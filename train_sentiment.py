from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "processed"
HF_CACHE_DIR = ROOT.parent / "HF-Cache"
BASE_MODEL = "distilbert/distilbert-base-multilingual-cased"

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
LANGUAGES = ["en", "fa"]
LANGUAGE_TO_ID = {language: index for index, language in enumerate(LANGUAGES)}

SEED = 42
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 2
GRAD_ACCUMULATION = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2.0e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LOG_EVERY_UPDATES = 50
DEFAULT_EPOCHS = 1
ROWS_PER_LANGUAGE = 10_549  # use the complete smaller-language train split
SMOKE_TRAIN_ROWS = 512
SMOKE_VALIDATION_ROWS = 400
SMOKE_MAX_UPDATES = 20

DEFAULT_OUTPUT_DIR = ROOT / "models" / "sentiment_multilingual_v1"
SMOKE_OUTPUT_DIR = ROOT / "models" / "sentiment_multilingual_smoke_v1"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\u064a", "\u06cc").replace("\u0649", "\u06cc")
    text = text.replace("\u0643", "\u06a9").replace("\u0640", "")
    text = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_language_split(language: str, split: str) -> pd.DataFrame:
    path = DATA_ROOT / f"sentiment_{language}" / f"{split}.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty dataset file: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} for {language} is missing columns: {missing}")

    # IMPORTANT: source_label_id is intentionally not used. Its numeric mapping
    # differs between TweetEval and SentiPers; the semantic `label` column is stable.
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.loc[df["text"].str.strip().ne("")].copy()

    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unexpected labels in {path}: {unknown}")
    if df.empty:
        raise ValueError(f"No usable rows found in {path}")

    df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)
    df["language"] = language
    df["source"] = "tweeteval" if language == "en" else "sentipers"
    df["split"] = split
    return df.reset_index(drop=True)


def load_combined_split(split: str) -> pd.DataFrame:
    frames = [load_language_split(language, split) for language in LANGUAGES]
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {split}: {len(combined):,} rows")
    for language in LANGUAGES:
        subset = combined[combined["language"] == language]
        print(
            f"  {language}: {len(subset):,} rows | "
            f"{subset['label'].value_counts().reindex(LABELS, fill_value=0).to_dict()}"
        )
    print("  combined labels:", combined["label"].value_counts().reindex(LABELS, fill_value=0).to_dict())
    return combined.reset_index(drop=True)


def remove_train_validation_overlap(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> pd.DataFrame:
    """Remove normalized text overlaps from TRAIN while preserving VALIDATION.

    The validation split is kept unchanged. Any training row whose normalized
    text also occurs in validation is excluded from this training run, preventing
    leakage without modifying the source CSV files.
    """
    indices_to_remove: list[int] = []
    report: dict[str, dict[str, int]] = {}

    for language in LANGUAGES:
        train_lang = train_df.loc[train_df["language"] == language]
        val_lang = val_df.loc[val_df["language"] == language]

        val_keys = val_lang["text"].map(normalize_text_key)
        val_keys = val_keys[val_keys.ne("")]
        val_key_set = set(val_keys)

        train_keys = train_lang["text"].map(normalize_text_key)
        overlap_mask = train_keys.isin(val_key_set) & train_keys.ne("")
        overlapping_rows = train_lang.loc[overlap_mask]

        # Count unique normalized texts separately from rows removed.
        overlapping_keys = set(train_keys.loc[overlap_mask])

        # Record whether an overlapping training row carries a label not present
        # for the same normalized text in validation. Validation remains untouched.
        val_labels_by_key: dict[str, set[str]] = {}
        for key, label in zip(
            val_lang["text"].map(normalize_text_key), val_lang["label"].astype(str)
        ):
            if key:
                val_labels_by_key.setdefault(key, set()).add(label)

        conflicting_label_rows = 0
        for key, label in zip(
            overlapping_rows["text"].map(normalize_text_key),
            overlapping_rows["label"].astype(str),
        ):
            if label not in val_labels_by_key.get(key, set()):
                conflicting_label_rows += 1

        indices_to_remove.extend(overlapping_rows.index.tolist())
        report[language] = {
            "unique_overlapping_texts": len(overlapping_keys),
            "training_rows_removed": len(overlapping_rows),
            "overlapping_rows_with_label_disagreement": conflicting_label_rows,
        }

    cleaned = train_df.drop(index=indices_to_remove).reset_index(drop=True)
    removed_total = len(indices_to_remove)

    print("Train/validation normalized-overlap handling (validation preserved):")
    for language in LANGUAGES:
        item = report[language]
        print(
            f"  {language}: {item['unique_overlapping_texts']} unique texts; "
            f"removed {item['training_rows_removed']} train rows; "
            f"label disagreements: {item['overlapping_rows_with_label_disagreement']}"
        )
    print(f"  Total training rows removed: {removed_total}")

    if cleaned.empty:
        raise ValueError("All training rows were removed while resolving train/validation overlap.")
    for language in LANGUAGES:
        if not (cleaned["language"] == language).any():
            raise ValueError(f"No training rows remain for language {language} after overlap removal.")

    # Final safety assertion: the remaining training split must be disjoint
    # from validation under the exact same normalization rule.
    for language in LANGUAGES:
        train_remaining = set(
            cleaned.loc[cleaned["language"] == language, "text"].map(normalize_text_key)
        )
        val_remaining = set(
            val_df.loc[val_df["language"] == language, "text"].map(normalize_text_key)
        )
        train_remaining.discard("")
        val_remaining.discard("")
        if train_remaining.intersection(val_remaining):
            raise RuntimeError(f"Overlap remains for language {language} after cleanup.")

    print("Verified: no normalized train/validation overlap remains.")
    return cleaned


def _allocate_stratified_quotas(groups: list[pd.DataFrame], target_rows: int) -> list[int]:
    sizes = np.asarray([len(group) for group in groups], dtype=int)
    if target_rows >= int(sizes.sum()):
        return sizes.tolist()
    if target_rows < len(groups):
        raise ValueError(
            f"Requested {target_rows} rows, but at least {len(groups)} are needed "
            "to represent all strata."
        )

    raw = sizes / sizes.sum() * target_rows
    quotas = np.floor(raw).astype(int)
    quotas = np.minimum(np.maximum(quotas, 1), sizes)

    while int(quotas.sum()) > target_rows:
        candidates = np.where(quotas > 1)[0]
        if len(candidates) == 0:
            raise ValueError("Could not allocate requested stratified sample size.")
        idx = candidates[np.argmin(raw[candidates] - quotas[candidates])]
        quotas[idx] -= 1
    while int(quotas.sum()) < target_rows:
        candidates = np.where(quotas < sizes)[0]
        if len(candidates) == 0:
            break
        idx = candidates[np.argmax(raw[candidates] - quotas[candidates])]
        quotas[idx] += 1
    return quotas.tolist()


def stratified_sample(
    df: pd.DataFrame,
    target_rows: int,
    stratify_columns: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    if target_rows >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    grouped = [group for _, group in df.groupby(list(stratify_columns), sort=True, dropna=False)]
    quotas = _allocate_stratified_quotas(grouped, target_rows)
    parts: list[pd.DataFrame] = []
    for index, (group, count) in enumerate(zip(grouped, quotas)):
        if count == len(group):
            parts.append(group.copy())
        else:
            parts.append(group.sample(n=count, random_state=seed + index * 997))
    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_balanced_train_frame(train_df: pd.DataFrame, rows_per_language: int, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for language in LANGUAGES:
        language_df = train_df[train_df["language"] == language].copy()
        requested = min(rows_per_language, len(language_df))
        if requested < len(language_df):
            print(
                f"  Downsampling {language}: {len(language_df):,} -> {requested:,} "
                "(stratified by sentiment label; source CSV remains unchanged)"
            )
            language_df = stratified_sample(language_df, requested, ["label"], seed + LANGUAGE_TO_ID[language] * 1009)
        else:
            language_df = language_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        parts.append(language_df)

    balanced = pd.concat(parts, ignore_index=True)
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def encode_frame(df: pd.DataFrame, tokenizer: Any) -> TensorDataset:
    enc = tokenizer(
        df["text"].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    labels = torch.tensor(df["label_id"].to_numpy(dtype=np.int64), dtype=torch.long)
    language_ids = torch.tensor(
        df["language"].map(LANGUAGE_TO_ID).to_numpy(dtype=np.int64), dtype=torch.long
    )
    return TensorDataset(enc["input_ids"], enc["attention_mask"], labels, language_ids)


def make_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["label"].value_counts().reindex(LABELS, fill_value=0).to_numpy(dtype=np.float32)
    if np.any(counts <= 0):
        absent = [LABELS[i] for i, count in enumerate(counts) if count <= 0]
        raise ValueError(f"Missing training examples for sentiment classes: {absent}")
    weights = len(train_df) / (len(LABELS) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    label_ids = list(range(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    per_class: dict[str, Any] = {}
    for idx, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(precision[idx]), 6),
            "recall": round(float(recall[idx]), 6),
            "f1": round(float(f1[idx]), 6),
            "support": int(support[idx]),
        }
    return {
        "rows": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)), 6),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
        "confusion_matrix_labels": LABELS,
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    loss_total = 0.0
    rows_total = 0
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    lang_parts: list[np.ndarray] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            input_ids, attention_mask, labels, language_ids = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels_device)
            count = int(labels.size(0))
            loss_total += float(loss.item()) * count
            rows_total += count
            y_true_parts.append(labels.numpy())
            y_pred_parts.append(torch.argmax(logits, dim=1).cpu().numpy())
            lang_parts.append(language_ids.numpy())
            if batch_index % 250 == 0 or batch_index == len(loader):
                print(f"  validation batches: {batch_index}/{len(loader)}")

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    lang_ids = np.concatenate(lang_parts)
    result: dict[str, Any] = {"overall": classification_metrics(y_true, y_pred), "by_language": {}}

    for language, language_id in LANGUAGE_TO_ID.items():
        mask = lang_ids == language_id
        if mask.any():
            result["by_language"][language] = classification_metrics(y_true[mask], y_pred[mask])

    language_macro_f1 = [
        result["by_language"][language]["macro_f1"]
        for language in LANGUAGES
        if language in result["by_language"]
    ]
    result["balanced_language_macro_f1"] = (
        round(float(np.mean(language_macro_f1)), 6) if language_macro_f1 else None
    )
    return loss_total / max(rows_total, 1), result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bilingual three-class sentiment classifier.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--rows-per-language",
        type=int,
        default=ROWS_PER_LANGUAGE,
        help="Maximum training rows selected per language (stratified by label).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.rows_per_language < 1:
        raise ValueError("--rows-per-language must be at least 1")

    output_dir = (args.output_dir or (SMOKE_OUTPUT_DIR if args.smoke_test else DEFAULT_OUTPUT_DIR)).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Choose another path with --output-dir to avoid overwriting a previous run."
        )

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("BILINGUAL SENTIMENT — THREE-CLASS SINGLE-LABEL TRAINING")
    print("=" * 72)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        torch.cuda.reset_peak_memory_stats(0)
    print(f"Base model: {BASE_MODEL}")
    print(f"HF cache: {HF_CACHE_DIR}")
    print(f"Output: {output_dir}")
    print("Task: single-label three-class classification (softmax + CrossEntropyLoss)")

    full_train_df = load_combined_split("train")
    full_val_df = load_combined_split("validation")
    full_train_df = remove_train_validation_overlap(full_train_df, full_val_df)

    train_df = make_balanced_train_frame(full_train_df, args.rows_per_language, SEED)
    val_df = full_val_df.copy()

    if args.smoke_test:
        train_df = stratified_sample(
            train_df, min(SMOKE_TRAIN_ROWS, len(train_df)), ["language", "label"], SEED + 11
        )
        val_df = stratified_sample(
            val_df, min(SMOKE_VALIDATION_ROWS, len(val_df)), ["language", "label"], SEED + 29
        )
        epochs = 1
        max_updates: int | None = SMOKE_MAX_UPDATES
        print("\nMODE: SMOKE TEST — sampled rows and capped optimizer updates; not a final model")
    else:
        epochs = args.epochs
        max_updates = None
        print("\nMODE: FULL TRAINING")

    print(f"Original train rows: {len(full_train_df):,}")
    print(f"Balanced train rows used: {len(train_df):,}")
    print("  by language:", train_df["language"].value_counts().to_dict())
    print("  by source:", train_df["source"].value_counts().to_dict())
    print("  label counts:", train_df["label"].value_counts().reindex(LABELS, fill_value=0).to_dict())
    print(f"Validation rows used: {len(val_df):,}")
    print("  by language:", val_df["language"].value_counts().to_dict())
    print("  label counts:", val_df["label"].value_counts().reindex(LABELS, fill_value=0).to_dict())
    print(
        f"Epochs: {epochs}; batch size: {TRAIN_BATCH_SIZE}; "
        f"gradient accumulation: {GRAD_ACCUMULATION}; "
        f"effective batch: {TRAIN_BATCH_SIZE * GRAD_ACCUMULATION}"
    )
    print(f"Max length: {MAX_LENGTH}; learning rate: {LEARNING_RATE:g}")
    print("Checkpoint selection: mean of English and Persian validation macro-F1")
    print("Neither English nor Persian test.csv / test_official.csv will be loaded.")
    print("Note: source_label_id is ignored; semantic string labels are mapped explicitly.")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("Install Transformers with: python -m pip install -U transformers") from exc

    local_only = not args.allow_download
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, cache_dir=str(HF_CACHE_DIR), use_fast=True, local_files_only=local_only
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=local_only,
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            problem_type="single_label_classification",
        )
    except OSError as exc:
        if local_only:
            raise RuntimeError(
                "The base model is not fully available in the local HF cache. "
                "Connect to the internet and rerun with --allow-download."
            ) from exc
        raise

    model.to(device)
    print(f"Total parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    print("\nTokenizing balanced training data...")
    train_dataset = encode_frame(train_df, tokenizer)
    print("Tokenizing validation data...")
    val_dataset = encode_frame(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(SEED),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    counts = train_df["label"].value_counts().reindex(LABELS, fill_value=0).to_dict()
    class_weights = torch.tensor(
        [len(train_df) / (len(LABELS) * counts[label]) for label in LABELS],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_weights / class_weights.mean()
    print("Class weights:", {label: round(float(weight), 4) for label, weight in zip(LABELS, class_weights.cpu())})

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUMULATION)
    actual_updates = min(updates_per_epoch, max_updates) if max_updates is not None else updates_per_epoch
    total_updates = actual_updates * epochs
    warmup_steps = max(1, int(total_updates * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )

    print(f"Optimizer updates/epoch: {updates_per_epoch}; total planned updates: {total_updates}")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    global_step = 0
    history: list[dict[str, Any]] = []
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        rows_seen = 0
        epoch_start = time.perf_counter()
        updates_this_epoch = 0

        for batch_index, batch in enumerate(train_loader):
            if max_updates is not None and updates_this_epoch >= max_updates:
                break
            input_ids, attention_mask, labels, _language_ids = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels_device)
            batch_rows = int(labels.size(0))
            loss_total += float(loss.item()) * batch_rows
            rows_seen += batch_rows
            (loss / GRAD_ACCUMULATION).backward()

            should_update = (
                (batch_index + 1) % GRAD_ACCUMULATION == 0
                or (batch_index + 1) == len(train_loader)
                or (
                    max_updates is not None
                    and updates_this_epoch + 1 == max_updates
                    and (batch_index + 1) % GRAD_ACCUMULATION != 0
                )
            )
            if should_update:
                remainder = (batch_index + 1) % GRAD_ACCUMULATION
                if remainder:
                    correction = GRAD_ACCUMULATION / remainder
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                updates_this_epoch += 1

                if global_step % LOG_EVERY_UPDATES == 0 or global_step == 1 or updates_this_epoch == actual_updates:
                    elapsed = time.perf_counter() - epoch_start
                    eta_minutes = max(actual_updates - updates_this_epoch, 0) * elapsed / max(updates_this_epoch, 1) / 60
                    print(
                        f"Epoch {epoch}/{epochs} | update {updates_this_epoch}/{actual_updates} "
                        f"| batch loss {loss.item():.4f} | ETA about {eta_minutes:.0f} min"
                    )

        train_loss = loss_total / max(rows_seen, 1)
        print("\nRunning validation...")
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        selection_score = float(val_metrics["balanced_language_macro_f1"] or 0.0)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_metrics": val_metrics,
                "selection_metric": "mean_en_fa_validation_macro_f1",
                "selection_score": selection_score,
                "epoch_seconds": epoch_seconds,
            }
        )

        print(f"\nEPOCH {epoch} SUMMARY")
        print(f"  train loss: {train_loss:.5f}")
        print(f"  validation loss: {val_loss:.5f}")
        print(f"  overall macro-F1: {val_metrics['overall']['macro_f1']:.5f}")
        for language in LANGUAGES:
            score = val_metrics["by_language"].get(language, {}).get("macro_f1", float("nan"))
            print(f"  {language} macro-F1: {score:.5f}")
        print(f"  balanced-language macro-F1: {selection_score:.5f}")
        print(f"  epoch runtime: {epoch_seconds / 60:.1f} minutes")

        if selection_score > best_score:
            best_score = selection_score
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            summary = {
                "base_model": BASE_MODEL,
                "task": "bilingual_three_class_sentiment_classification",
                "labels": LABELS,
                "label_to_id": LABEL_TO_ID,
                "max_length": MAX_LENGTH,
                "training_rows_after_overlap_removal": len(full_train_df),
                "training_rows_used": len(train_df),
                "training_rows_by_language": train_df["language"].value_counts().to_dict(),
                "original_training_rows_by_language": full_train_df["language"].value_counts().to_dict(),
                "training_rows_by_source": train_df["source"].value_counts().to_dict(),
                "training_class_counts": train_df["label"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
                "validation_rows": len(val_df),
                "validation_rows_by_language": val_df["language"].value_counts().to_dict(),
                "validation_class_counts": val_df["label"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
                "epochs_requested": epochs,
                "best_epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_metrics": val_metrics,
                "class_weights": {label: float(weight) for label, weight in zip(LABELS, class_weights.cpu())},
                "learning_rate": LEARNING_RATE,
                "sampling_strategy": f"stratified maximum of {args.rows_per_language} rows per language; semantic labels used; source_label_id ignored",
                "selection_metric": "mean_en_fa_validation_macro_f1",
                "smoke_test": bool(args.smoke_test),
                "english_test_used": False,
                "persian_test_used": False,
            }
            with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            print(f"  Saved best checkpoint: {output_dir}")

        with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump({"history": history, "best_balanced_language_macro_f1": best_score}, handle, ensure_ascii=False, indent=2)

    total_seconds = time.perf_counter() - training_start
    if device.type == "cuda":
        print(f"\nPeak GPU memory allocated: {torch.cuda.max_memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"Total runtime: {total_seconds / 60:.1f} minutes")
    print(f"Best checkpoint: {output_dir}")
    print("Done. No test.csv or test_official.csv files were loaded or modified.")


if __name__ == "__main__":
    main()
