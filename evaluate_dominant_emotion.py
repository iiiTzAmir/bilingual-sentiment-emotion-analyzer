from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "emotion_dominant_v1"
CHECKPOINT_DIR = ROOT / "models" / "emotion_dominant_v1"
OUTPUT_DIR = ROOT / "reports" / "emotion_dominant_v1"
LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
REQUIRED_COLUMNS = ["text", "gold_label", "label_id", "source", "language"]


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, Any]):
        self.encodings = encodings

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: value[index]
            for key, value in self.encodings.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }


def load_dataset(path: Path, expected_split: str | None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty dataset file: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    frame = frame.copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    if frame["text"].str.strip().eq("").any():
        raise ValueError(f"{path.name} contains empty text rows")

    frame["gold_label"] = frame["gold_label"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(frame["gold_label"]) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected labels in {path.name}: {unexpected}")

    expected_ids = frame["gold_label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    numeric_ids = pd.to_numeric(frame["label_id"], errors="coerce")
    if numeric_ids.isna().any() or not np.array_equal(
        numeric_ids.to_numpy(dtype=np.int64), expected_ids
    ):
        raise ValueError(f"label_id does not match gold_label in {path.name}")
    frame["label_id"] = expected_ids

    frame["language"] = frame["language"].astype(str).str.strip().str.lower()
    frame["source"] = frame["source"].astype(str).str.strip().str.lower()
    if not set(frame["language"]).issubset({"en", "fa"}):
        raise ValueError(f"Unexpected language values in {path.name}: {sorted(set(frame['language']))}")
    if expected_split and "split" in frame.columns:
        actual = set(frame["split"].astype(str).str.strip().str.lower())
        if actual != {expected_split}:
            raise ValueError(
                f"Expected split={expected_split!r} in {path.name}, found {sorted(actual)}"
            )
    return frame.reset_index(drop=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    label_ids = list(range(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    per_class: dict[str, Any] = {}
    for index, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(precision[index]), 6), # type: ignore
            "recall": round(float(recall[index]), 6), # type: ignore
            "f1": round(float(f1[index]), 6), # type: ignore
            "support": int(support[index]), # type: ignore
        }

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    return {
        "rows": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)), 6),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_labels": LABELS,
    }


def evaluate_groups(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ("language", "source"):
        grouped: dict[str, Any] = {}
        for key, indices in frame.groupby(column, sort=True).indices.items():
            idx = np.asarray(indices, dtype=np.int64)
            grouped[str(key)] = calculate_metrics(y_true[idx], y_pred[idx])
        result[f"by_{column}"] = grouped

    # Equal language weighting prevents the larger language partition from dominating.
    language_f1 = [
        group_metrics["macro_f1"]
        for group_metrics in result["by_language"].values()
        if group_metrics["rows"] > 0
    ]
    result["balanced_language_macro_f1"] = (
        round(float(np.mean(language_f1)), 6) if language_f1 else None
    )
    return result


def predict_frame(
    frame: pd.DataFrame,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    texts = frame["text"].astype(str).tolist()
    start = time.perf_counter()
    model.eval()
    logits_list: list[np.ndarray] = []

    with torch.inference_mode():
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index : start_index + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            logits = model(**model_inputs).logits
            logits_list.append(logits.detach().cpu().numpy())

            done = min(start_index + len(batch_texts), len(texts))
            if done % 1000 < batch_size or done == len(texts):
                print(f"  inference: {done:,}/{len(texts):,}")

    logits_array = np.concatenate(logits_list, axis=0)
    # Stable softmax in float64 on CPU for report probabilities.
    shifted = logits_array.astype(np.float64) - logits_array.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    elapsed = time.perf_counter() - start
    return predictions, probabilities, elapsed


def save_predictions(frame: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    output = frame.copy()
    output["predicted_label_id"] = predictions
    output["predicted_label"] = [ID_TO_LABEL[int(index)] for index in predictions]
    output["correct"] = predictions == output["label_id"].to_numpy(dtype=np.int64)
    output["predicted_confidence"] = probabilities.max(axis=1).round(6)
    for index, label in enumerate(LABELS):
        output[f"prob_{label}"] = probabilities[:, index].round(6)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions: {path}")


def evaluate_one(
    name: str,
    path: Path,
    expected_split: str | None,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    max_length: int,
    output_dir: Path,
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    frame = load_dataset(path, expected_split)
    print(f"Rows: {len(frame):,}")
    print("By language:", frame["language"].value_counts().to_dict())
    print("By source:", frame["source"].value_counts().to_dict())
    print("Gold-label counts:", frame["gold_label"].value_counts().reindex(LABELS, fill_value=0).to_dict())

    predictions, probabilities, elapsed = predict_frame(
        frame, tokenizer, model, device, batch_size, max_length
    )
    y_true = frame["label_id"].to_numpy(dtype=np.int64)
    metrics = calculate_metrics(y_true, predictions)
    metrics.update(evaluate_groups(frame, y_true, predictions))
    metrics["inference_seconds"] = round(elapsed, 2)
    metrics["device"] = str(device)
    metrics["max_length"] = max_length
    metrics["batch_size"] = batch_size

    prediction_path = output_dir / f"{name.lower().replace(' ', '_')}_predictions.csv"
    save_predictions(frame, predictions, probabilities, prediction_path)

    print("\nMetrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return {
        "dataset_name": name,
        "dataset_path": str(path),
        "metrics": metrics,
        "predictions_path": str(prediction_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen dominant-emotion model on internal and external test sets."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Defaults to CPU to avoid additional GPU load.",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Model checkpoint directory not found: {checkpoint_dir}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_length < 8:
        raise ValueError("--max-length must be at least 8")

    # Explicit CPU default. CUDA is used only if the user passes --device cuda.
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    device = torch.device(args.device)

    main_test_path = data_dir / "test.csv"
    external_test_path = data_dir / "armanemo_external_test.csv"
    for required_path in (main_test_path, external_test_path):
        if not required_path.exists() or required_path.stat().st_size == 0:
            raise FileNotFoundError(f"Required test file is missing or empty: {required_path}")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install Transformers with: python -m pip install -U transformers") from exc

    print("=" * 72)
    print("FINAL DOMINANT-EMOTION EVALUATION")
    print("=" * 72)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("GPU will not be used.")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {args.batch_size}; max length: {args.max_length}")
    print("Test sets are evaluated only; no training or weight updates will occur.")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )
    model.to(device)

    config_id2label = getattr(model.config, "id2label", {}) or {}
    if config_id2label:
        saved_labels = [str(config_id2label.get(index, config_id2label.get(str(index), ""))).lower() for index in range(len(LABELS))]
        if saved_labels != LABELS:
            raise ValueError(
                f"Checkpoint label order {saved_labels} does not match expected order {LABELS}. "
                "Do not evaluate with a mismatched label mapping."
            )
    if int(model.config.num_labels) != len(LABELS):
        raise ValueError(f"Checkpoint has {model.config.num_labels} outputs; expected {len(LABELS)}")

    print(f"Model outputs: {model.config.num_labels}")
    print("Label order:", LABELS)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.perf_counter()
    main_result = evaluate_one(
        "Main Test",
        main_test_path,
        "test",
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
        output_dir,
    )
    external_result = evaluate_one(
        "ArmanEmo External Test",
        external_test_path,
        None,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
        output_dir,
    )
    total_seconds = time.perf_counter() - start_all

    report = {
        "checkpoint": str(checkpoint_dir),
        "device": str(device),
        "labels": LABELS,
        "task": "single_label_five_class_dominant_emotion",
        "evaluation_only": True,
        "training_performed": False,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "main_test": main_result,
        "armanemo_external_test": external_result,
        "total_evaluation_seconds": round(total_seconds, 2),
        "interpretation_notes": [
            "Main Test is the held-out internal test split from GoEmotions and EmoPars.",
            "ArmanEmo External Test is a separate Persian cross-dataset evaluation and should be reported separately, not pooled with Main Test.",
            "Do not tune thresholds, labels, or hyperparameters on either test set; use validation for model selection.",
            "Results reflect the project's five-class dominant-label mapping, including the approximate Wonder-to-surprise mapping and source-specific annotation conventions.",
        ],
    }
    report_path = output_dir / "final_emotion_evaluation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 72)
    print("Main Test macro-F1:", main_result["metrics"]["macro_f1"])
    print("Main Test balanced-language macro-F1:", main_result["metrics"]["balanced_language_macro_f1"])
    print("ArmanEmo External Test macro-F1:", external_result["metrics"]["macro_f1"])
    print("ArmanEmo External Test balanced-language macro-F1:", external_result["metrics"]["balanced_language_macro_f1"])
    print(f"Total evaluation time: {total_seconds / 60:.1f} minutes")
    print(f"Saved final report: {report_path}")
    print("Done. No training was performed; model weights and test CSV files were not modified.")


if __name__ == "__main__":
    main()
