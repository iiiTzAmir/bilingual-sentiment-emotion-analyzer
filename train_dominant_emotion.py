from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "emotion_dominant_v1"
HF_CACHE_DIR = ROOT.parent / "HF-Cache"
BASE_MODEL = "distilbert/distilbert-base-multilingual-cased"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "emotion_dominant_v1"
SMOKE_OUTPUT_DIR = ROOT / "models" / "emotion_dominant_smoke_v1"

LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
LANGUAGES = ["en", "fa"]
SOURCES = ["goemotions", "emopars", "armanemo"]
LANGUAGE_TO_ID = {name: index for index, name in enumerate(LANGUAGES)}
SOURCE_TO_ID = {name: index for index, name in enumerate(SOURCES)}

SEED = 42
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 2
GRAD_ACCUMULATION = 8
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 2.0e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LOG_EVERY_UPDATES = 50
DEFAULT_EPOCHS = 2
SMOKE_TRAIN_ROWS = 512
SMOKE_VALIDATION_ROWS = 400
SMOKE_MAX_UPDATES = 20


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\u064a", "\u06cc").replace("\u0649", "\u06cc")
    text = text.replace("\u0643", "\u06a9").replace("\u0640", "")
    text = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty dataset file: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["text", "gold_label", "label_id", "source", "language", "split"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name}.csv is missing required columns: {missing}")

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    if df["text"].str.strip().eq("").any():
        raise ValueError(f"{name}.csv contains empty text")

    df["gold_label"] = df["gold_label"].astype(str).str.strip().str.lower()
    unknown_labels = sorted(set(df["gold_label"]) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected labels in {name}.csv: {unknown_labels}")
    expected_ids = df["gold_label"].map(LABEL_TO_ID).astype(int)
    actual_ids = pd.to_numeric(df["label_id"], errors="coerce")
    if actual_ids.isna().any() or not np.array_equal(actual_ids.to_numpy(dtype=int), expected_ids.to_numpy()):
        raise ValueError(f"label_id does not match gold_label in {name}.csv")
    df["label_id"] = expected_ids

    df["language"] = df["language"].astype(str).str.lower().str.strip()
    unknown_languages = sorted(set(df["language"]) - set(LANGUAGES))
    if unknown_languages:
        raise ValueError(f"Unexpected languages in {name}.csv: {unknown_languages}")
    df["source"] = df["source"].astype(str).str.lower().str.strip()
    unknown_sources = sorted(set(df["source"]) - set(SOURCES))
    if unknown_sources:
        raise ValueError(f"Unexpected sources in {name}.csv: {unknown_sources}")

    if not (df["split"].astype(str).str.lower() == name).all():
        raise ValueError(f"The split column in {name}.csv contains unexpected values")

    print(f"Loaded {name}: {len(df):,} rows")
    print("  by language:", df["language"].value_counts().to_dict())
    print("  by source:", df["source"].value_counts().to_dict())
    print("  class counts:", df["gold_label"].value_counts().reindex(LABELS, fill_value=0).to_dict())
    return df.reset_index(drop=True)


def validate_no_train_validation_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    train_keys = set(train_df["text"].map(normalize_key))
    val_keys = set(val_df["text"].map(normalize_key))
    train_keys.discard("")
    val_keys.discard("")
    overlap = train_keys.intersection(val_keys)
    if overlap:
        raise ValueError(
            f"Found {len(overlap)} normalized text(s) shared by train and validation. "
            "Re-run prepare_dominant_emotion.py before training."
        )
    print("Train/validation text-overlap check: 0")


def stratified_sample(df: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    """Deterministic proportional sample stratified by language and class."""
    if target_rows >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if target_rows < df[["language", "gold_label"]].drop_duplicates().shape[0]:
        raise ValueError("Smoke-test sample is too small to represent every language/class group")

    grouped = list(df.groupby(["language", "gold_label"], sort=True))
    sizes = np.array([len(group) for _, group in grouped], dtype=int)
    raw = sizes / sizes.sum() * target_rows
    quotas = np.floor(raw).astype(int)
    quotas = np.maximum(quotas, 1)

    while quotas.sum() > target_rows:
        candidates = np.where(quotas > 1)[0]
        if len(candidates) == 0:
            break
        idx = candidates[np.argmin(raw[candidates] - quotas[candidates])]
        quotas[idx] -= 1
    while quotas.sum() < target_rows:
        candidates = np.where(quotas < sizes)[0]
        if len(candidates) == 0:
            break
        idx = candidates[np.argmax(raw[candidates] - quotas[candidates])]
        quotas[idx] += 1

    parts = []
    for (key, group), count in zip(grouped, quotas):
        parts.append(group.sample(n=int(count), random_state=seed + sum(map(ord, str(key)))))
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def encode_frame(df: pd.DataFrame, tokenizer: Any) -> TensorDataset:
    encoded = tokenizer(
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
    source_ids = torch.tensor(
        df["source"].map(SOURCE_TO_ID).to_numpy(dtype=np.int64), dtype=torch.long
    )
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels, language_ids, source_ids)


def make_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["gold_label"].value_counts().reindex(LABELS, fill_value=0).to_numpy(dtype=np.float32)
    if np.any(counts <= 0):
        absent = [LABELS[i] for i, value in enumerate(counts) if value <= 0]
        raise ValueError(f"No training examples for classes: {absent}")
    weights = len(train_df) / (len(LABELS) * counts)
    # Normalize to mean 1 so the overall loss scale remains stable.
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(range(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for idx, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(precision[idx]), 6), # type: ignore
            "recall": round(float(recall[idx]), 6), # type: ignore
            "f1": round(float(f1[idx]), 6), # type: ignore
            "support": int(support[idx]), # type: ignore
        }
    return {
        "rows": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 6),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    eval_df: pd.DataFrame,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    loss_sum = 0.0
    row_count = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for batch_index, (input_ids, attention_mask, labels, _language_ids, _source_ids) in enumerate(loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels_device)
            batch_rows = int(labels.size(0))
            loss_sum += float(loss.item()) * batch_rows
            row_count += batch_rows
            all_true.append(labels.numpy())
            all_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
            if batch_index % 250 == 0 or batch_index == len(loader):
                print(f"  validation batches: {batch_index}/{len(loader)}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    result: dict[str, Any] = {"overall": classification_metrics(y_true, y_pred), "by_language": {}, "by_source": {}}

    languages = eval_df["language"].to_numpy()
    sources = eval_df["source"].to_numpy()
    for language in LANGUAGES:
        mask = languages == language
        if mask.any():
            result["by_language"][language] = classification_metrics(y_true[mask], y_pred[mask])
    for source in SOURCES:
        mask = sources == source
        if mask.any():
            result["by_source"][source] = classification_metrics(y_true[mask], y_pred[mask])

    language_f1 = [result["by_language"][language]["macro_f1"] for language in LANGUAGES if language in result["by_language"]]
    result["balanced_language_macro_f1"] = round(float(np.mean(language_f1)), 6) if language_f1 else None
    return loss_sum / max(row_count, 1), result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the five-class dominant-emotion classifier.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run a small pipeline check; not a usable model.")
    parser.add_argument("--allow-download", action="store_true", help="Allow fetching the base model if it is not cached locally.")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    output_dir = (args.output_dir or (SMOKE_OUTPUT_DIR if args.smoke_test else DEFAULT_OUTPUT_DIR)).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Choose a new path with --output-dir to avoid overwriting a previous run."
        )

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("DOMINANT EMOTION — FIVE-CLASS SINGLE-LABEL TRAINING")
    print("=" * 72)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        torch.cuda.reset_peak_memory_stats(0)
    print(f"Base model: {BASE_MODEL}")
    print(f"HF cache: {HF_CACHE_DIR}")
    print(f"Output: {output_dir}")
    print("Task: single-label five-class classification (softmax + CrossEntropyLoss)")

    train_df = load_split("train")
    val_df = load_split("validation")
    validate_no_train_validation_overlap(train_df, val_df)

    if args.smoke_test:
        train_df = stratified_sample(train_df, min(SMOKE_TRAIN_ROWS, len(train_df)), SEED)
        val_df = stratified_sample(val_df, min(SMOKE_VALIDATION_ROWS, len(val_df)), SEED + 1)
        epochs = 1
        max_updates = SMOKE_MAX_UPDATES
        print("\nMODE: SMOKE TEST — sampled rows and capped optimizer updates; not a final model")
    else:
        epochs = args.epochs
        max_updates = None
        print("\nMODE: FULL TRAINING")

    print(f"Training rows used: {len(train_df):,}")
    print("  language counts:", train_df["language"].value_counts().to_dict())
    print("  source counts:", train_df["source"].value_counts().to_dict())
    print("  class counts:", train_df["gold_label"].value_counts().reindex(LABELS, fill_value=0).to_dict())
    print(f"Validation rows used: {len(val_df):,}")
    print("  language counts:", val_df["language"].value_counts().to_dict())
    print("  source counts:", val_df["source"].value_counts().to_dict())
    print(f"Epochs: {epochs}; batch size: {TRAIN_BATCH_SIZE}; gradient accumulation: {GRAD_ACCUMULATION}; effective batch: {TRAIN_BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"Max length: {MAX_LENGTH}; learning rate: {LEARNING_RATE:g}")
    print("Checkpoint selection: mean of English and Persian validation macro-F1")
    print("Neither main test.csv nor armanemo_external_test.csv will be loaded.")

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
    print("\nTokenizing training data...")
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

    class_weights = make_class_weights(train_df, device)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    global_step = 0
    history: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        weighted_loss_sum = 0.0
        rows_seen = 0
        epoch_start = time.perf_counter()
        updates_this_epoch = 0

        for batch_index, batch in enumerate(train_loader):
            if max_updates is not None and updates_this_epoch >= max_updates:
                break
            input_ids, attention_mask, labels, _language_ids, _source_ids = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels_device)
            batch_rows = int(labels.size(0))
            weighted_loss_sum += float(loss.item()) * batch_rows
            rows_seen += batch_rows
            (loss / GRAD_ACCUMULATION).backward()

            should_update = (
                (batch_index + 1) % GRAD_ACCUMULATION == 0
                or (batch_index + 1) == len(train_loader)
                or (max_updates is not None and updates_this_epoch + 1 == max_updates and (batch_index + 1) % GRAD_ACCUMULATION != 0)
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
                    eta = max(actual_updates - updates_this_epoch, 0) * elapsed / max(updates_this_epoch, 1) / 60
                    print(
                        f"Epoch {epoch}/{epochs} | update {updates_this_epoch}/{actual_updates} "
                        f"| batch loss {loss.item():.4f} | ETA about {eta:.0f} min"
                    )

        train_loss = weighted_loss_sum / max(rows_seen, 1)
        print("\nRunning full validation...")
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, val_df)
        selection_score = float(val_metrics["balanced_language_macro_f1"] or 0.0)
        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_metrics": val_metrics,
            "selection_metric": "mean_en_fa_validation_macro_f1",
            "selection_score": selection_score,
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)

        print(f"\nEPOCH {epoch} SUMMARY")
        print(f"  train loss: {train_loss:.5f}")
        print(f"  validation loss: {val_loss:.5f}")
        print(f"  overall macro-F1: {val_metrics['overall']['macro_f1']:.5f}")
        print(f"  English macro-F1: {val_metrics['by_language'].get('en', {}).get('macro_f1', float('nan')):.5f}")
        print(f"  Persian macro-F1: {val_metrics['by_language'].get('fa', {}).get('macro_f1', float('nan')):.5f}")
        print(f"  balanced-language macro-F1: {selection_score:.5f}")
        print("  by-source macro-F1:", {source: metrics["macro_f1"] for source, metrics in val_metrics["by_source"].items()})
        print(f"  epoch runtime: {epoch_seconds / 60:.1f} minutes")

        if selection_score > best_score:
            best_score = selection_score
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            summary = {
                "base_model": BASE_MODEL,
                "initialization": "fresh_pretrained_base_model_not_previous_finetuned_checkpoint",
                "task": "single_label_five_class_dominant_emotion",
                "labels": LABELS,
                "label_to_id": LABEL_TO_ID,
                "max_length": MAX_LENGTH,
                "training_rows": len(train_df),
                "training_rows_by_language": train_df["language"].value_counts().to_dict(),
                "training_rows_by_source": train_df["source"].value_counts().to_dict(),
                "training_class_counts": train_df["gold_label"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
                "validation_rows": len(val_df),
                "validation_rows_by_language": val_df["language"].value_counts().to_dict(),
                "validation_rows_by_source": val_df["source"].value_counts().to_dict(),
                "epochs_requested": epochs,
                "best_epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_metrics": val_metrics,
                "class_weights": {label: float(weight) for label, weight in zip(LABELS, class_weights.cpu())},
                "learning_rate": LEARNING_RATE,
                "selection_metric": "mean_en_fa_validation_macro_f1",
                "smoke_test": bool(args.smoke_test),
                "main_test_used": False,
                "armanemo_external_test_used": False,
                "note": "Dominant means one active label among the project's five shared classes; original non-core co-labels may remain in source annotations.",
            }
            with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            print(f"  Saved best checkpoint: {output_dir}")

        with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump({"history": history, "best_balanced_language_macro_f1": best_score}, handle, ensure_ascii=False, indent=2)

    total_seconds = time.perf_counter() - total_start
    if device.type == "cuda":
        print(f"\nPeak GPU memory allocated: {torch.cuda.max_memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"Total runtime: {total_seconds / 60:.1f} minutes")
    print(f"Best checkpoint: {output_dir}")
    print("Done. Main test.csv and armanemo_external_test.csv were not loaded or modified.")


if __name__ == "__main__":
    main()
