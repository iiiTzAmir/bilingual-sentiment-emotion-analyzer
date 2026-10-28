"""Evaluate the trained bilingual sentiment model on clean, held-out test sets.

Run from the project root (or keep this file in the project root):
    python evaluate_sentiment.py

This script uses CPU only. It does not train or modify the model or datasets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def calculate_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "rows": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]), # type: ignore
                "recall": float(report[label]["recall"]), # type: ignore
                "f1": float(report[label]["f1-score"]), # type: ignore
                "support": int(report[label]["support"]), # type: ignore
            }
            for label in LABELS
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABELS,
    }


def predict_texts(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    max_length: int,
    language: str,
) -> tuple[list[int], list[float]]:
    all_predictions: list[int] = []
    all_confidences: list[float] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # Deliberately keep every tensor on CPU.
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidences, predictions = torch.max(probabilities, dim=-1)

            all_predictions.extend(predictions.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

            completed = min(start + len(batch_texts), len(texts))
            print(f"[{language}] {completed:,}/{len(texts):,} rows evaluated")

    return all_predictions, all_confidences


def evaluate_language(
    language: str,
    csv_path: Path,
    tokenizer: Any,
    model: Any,
    output_dir: Path,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[int], list[int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Test file not found: {csv_path}\n"
            "Check that evaluate_sentiment.py is in the project root and that the processed data folders exist."
        )

    df = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    # Keep only rows with usable text and one of the three semantic labels.
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    before = len(df)
    df = df[(df["text"] != "") & (df["label"].isin(LABEL_TO_ID))].reset_index(drop=True)
    excluded = before - len(df)
    if df.empty:
        raise ValueError(f"No valid test rows found in {csv_path}")

    y_true = [LABEL_TO_ID[label] for label in df["label"].tolist()]
    y_pred, confidence = predict_texts(
        df["text"].tolist(), tokenizer, model, batch_size, max_length, language
    )

    metrics = calculate_metrics(y_true, y_pred)
    metrics["excluded_invalid_rows"] = excluded
    metrics["test_file"] = str(csv_path)

    predictions_df = df.copy()
    predictions_df["predicted_label"] = [ID_TO_LABEL[value] for value in y_pred]
    predictions_df["confidence"] = confidence
    predictions_df["correct"] = [actual == predicted for actual, predicted in zip(y_true, y_pred)]

    output_path = output_dir / f"{language}_test_predictions.csv"
    predictions_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[{language}] saved predictions: {output_path}")

    return metrics, predictions_df, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only evaluation for the bilingual sentiment model.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be at least 1")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "models" / "sentiment_multilingual_v1"
    test_files = {
        "en": project_root / "data" / "processed" / "sentiment_en" / "test.csv",
        "fa": project_root / "data" / "processed" / "sentiment_fa" / "test.csv",
    }
    output_dir = project_root / "reports" / "sentiment_multilingual_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Trained model folder not found: {model_dir}")

    # Explicit CPU configuration: no CUDA context or GPU inference is used.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Model: {model_dir}")
    print(f"Batch size: {args.batch_size} | Max length: {args.max_length}")
    print("Using clean test.csv files only; test_official.csv is not loaded.\n")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)

    if model.config.num_labels != len(LABELS):
        raise ValueError(
            f"Expected {len(LABELS)} output classes, but the model has {model.config.num_labels}."
        )

    language_results: dict[str, Any] = {}
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for language in ("en", "fa"):
        metrics, _, y_true, y_pred = evaluate_language(
            language=language,
            csv_path=test_files[language],
            tokenizer=tokenizer,
            model=model,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        language_results[language] = metrics
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    overall_metrics = calculate_metrics(all_y_true, all_y_pred)
    balanced_language_macro_f1 = float(
        (language_results["en"]["macro_f1"] + language_results["fa"]["macro_f1"]) / 2
    )

    results = {
        "task": "bilingual_three_class_sentiment_classification",
        "model_dir": str(model_dir),
        "device": "cpu",
        "labels": LABELS,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "overall_combined": overall_metrics,
        "by_language": language_results,
        "balanced_language_macro_f1": balanced_language_macro_f1,
        "evaluation_note": (
            "English and Persian clean test.csv files were evaluated separately. "
            "The combined metric is row-weighted by the combined test-set size; "
            "balanced_language_macro_f1 is the arithmetic mean of the two language-specific macro-F1 scores. "
            "No training, validation, or test_official.csv data was used by this evaluation script."
        ),
    }

    json_path = output_dir / "final_sentiment_evaluation.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"English Macro-F1: {language_results['en']['macro_f1']:.4f}")
    print(f"Persian  Macro-F1: {language_results['fa']['macro_f1']:.4f}")
    print(f"Balanced-language Macro-F1: {balanced_language_macro_f1:.4f}")
    print(f"Combined Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Combined Macro-F1: {overall_metrics['macro_f1']:.4f}")
    print("\nPer-language class F1:")
    for language in ("en", "fa"):
        class_f1 = language_results[language]["per_class"]
        print(
            f"{language.upper()}: "
            + " | ".join(f"{label}={class_f1[label]['f1']:.4f}" for label in LABELS)
        )
    print(f"\nFull report saved to: {json_path}")
    print(f"Predictions saved in: {output_dir}")


if __name__ == "__main__":
    main()
