from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dominant-emotion dataset preparation
# Expected project layout (paths can be overridden with --root):
#   data/raw/goemotions/{train,validation,test}.csv
#   data/raw/emopars.csv OR the extracted EmoPars source CSV
#   data/raw/emotion_fa/_sources/armanemo/.../dataset/{train,test}.tsv
# Output:
#   data/processed/emotion_dominant_v1/
# ---------------------------------------------------------------------------

SEED = 42
LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABELS)}
EMOPARS_THRESHOLD = 3

GOEMOTIONS_LABEL_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]
GOEMOTIONS_CORE = set(LABELS)

ARMANEMO_ALL_LABELS = {"ANGRY", "FEAR", "HAPPY", "HATE", "OTHER", "SAD", "SURPRISE"}
ARMANEMO_TO_SHARED = {
    "ANGRY": "anger",
    "FEAR": "fear",
    "HAPPY": "joy",
    "SAD": "sadness",
    "SURPRISE": "surprise",
}

OUTPUT_COLUMNS = [
    "text", "gold_label", "label_id", "source", "language", "split", "original_labels"
]


def normalize_key(value: Any) -> str:
    """Conservative duplicate key; original text is always preserved for modeling."""
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    # Canonicalize common Arabic/Persian variants for duplicate detection only.
    text = text.replace("\u064a", "\u06cc").replace("\u0649", "\u06cc")
    text = text.replace("\u0643", "\u06a9").replace("\u0640", "")
    text = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def canonical_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def find_column(df: pd.DataFrame, *candidates: str) -> str:
    lookup = {canonical_column_name(column): str(column) for column in df.columns}
    for candidate in candidates:
        key = canonical_column_name(candidate)
        if key in lookup:
            return lookup[key]
    raise ValueError(
        f"Could not find columns {candidates}. Available columns: {list(df.columns)}"
    )


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_label_value(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return list(value)
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, float) and value.is_integer():
        return [int(value)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
            if isinstance(parsed, (int, np.integer)):
                return [int(parsed)]
            if isinstance(parsed, str):
                return [parsed]
        except (ValueError, SyntaxError):
            pass
        return [part.strip().strip("'\"[]") for part in re.split(r"[,;|]", stripped) if part.strip()]
    return [value]


def canonical_go_label(value: Any) -> str:
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, str) and value.strip().isdigit()
    ):
        index = int(value)
        if index < 0 or index >= len(GOEMOTIONS_LABEL_NAMES):
            raise ValueError(f"GoEmotions label ID {index} is out of range")
        return GOEMOTIONS_LABEL_NAMES[index]
    name = str(value).strip().casefold().replace(" ", "_")
    if name.isdigit():
        index = int(name)
        if index < 0 or index >= len(GOEMOTIONS_LABEL_NAMES):
            raise ValueError(f"GoEmotions label ID {index} is out of range")
        return GOEMOTIONS_LABEL_NAMES[index]
    return name


def build_row(
    text: Any,
    label: str,
    source: str,
    language: str,
    split: str,
    original_labels: str,
) -> dict[str, Any]:
    cleaned_text = "" if text is None or pd.isna(text) else str(text)
    return {
        "text": cleaned_text,
        "gold_label": label,
        "label_id": LABEL_TO_ID[label],
        "source": source,
        "language": language,
        "split": split,
        "original_labels": original_labels,
    }


def load_goemotions(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_dir = root / "data" / "raw" / "goemotions"
    selected_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"input_rows_by_split": {}, "selected_single_core_by_split": {},
                             "excluded_zero_core_by_split": {}, "excluded_multi_core_by_split": {},
                             "excluded_neutral_combo_by_split": {}, "excluded_empty_text_by_split": {}}

    for split in ["train", "validation", "test"]:
        path = input_dir / f"{split}.csv"
        df = read_csv_checked(path)
        text_col = find_column(df, "text")
        try:
            labels_col = find_column(df, "labels", "label")
        except ValueError:
            labels_col = None

        separate_cols = {
            label: {canonical_column_name(column): str(column) for column in df.columns}.get(
                canonical_column_name(label)
            )
            for label in GOEMOTIONS_LABEL_NAMES
        }
        separate_complete = all(separate_cols[label] is not None for label in GOEMOTIONS_LABEL_NAMES)
        if labels_col is None and not separate_complete:
            raise ValueError(
                f"{path} needs a `labels` column or all 28 emotion columns; found {list(df.columns)}"
            )

        counts = Counter()
        audit["input_rows_by_split"][split] = int(len(df))
        for _, row in df.iterrows():
            text = "" if pd.isna(row[text_col]) else str(row[text_col])
            if not normalize_key(text):
                counts["empty_text"] += 1
                continue

            if labels_col is not None:
                raw = parse_label_value(row[labels_col])
                # Some data exports store a 28-element multi-hot vector.
                is_vector = (
                    len(raw) == len(GOEMOTIONS_LABEL_NAMES)
                    and all(str(item).strip() in {"0", "1", "0.0", "1.0"} for item in raw)
                )
                if is_vector:
                    labels = {
                        name for name, active in zip(GOEMOTIONS_LABEL_NAMES, raw)
                        if int(float(active)) == 1
                    }
                else:
                    labels = {canonical_go_label(item) for item in raw}
            else:
                labels = set()
                for name, column in separate_cols.items():
                    try:
                        if float(row[column]) > 0: # type: ignore
                            labels.add(name)
                    except (TypeError, ValueError):
                        continue

            if "neutral" in labels and len(labels) > 1:
                counts["neutral_combo"] += 1
                continue

            core = labels.intersection(GOEMOTIONS_CORE)
            if len(core) == 0:
                counts["zero_core"] += 1
                continue
            if len(core) > 1:
                counts["multi_core"] += 1
                continue

            gold = next(iter(core))
            selected_rows.append(build_row(
                text=text,
                label=gold,
                source="goemotions",
                language="en",
                split=split,
                original_labels="|".join(sorted(labels)),
            ))
            counts["selected"] += 1
            if len(labels - core) > 0:
                counts["with_non_core_co_labels"] += 1

        audit["selected_single_core_by_split"][split] = int(counts["selected"])
        audit["excluded_zero_core_by_split"][split] = int(counts["zero_core"])
        audit["excluded_multi_core_by_split"][split] = int(counts["multi_core"])
        audit["excluded_neutral_combo_by_split"][split] = int(counts["neutral_combo"])
        audit["excluded_empty_text_by_split"][split] = int(counts["empty_text"])
        audit.setdefault("selected_with_non_core_co_labels_by_split", {})[split] = int(
            counts["with_non_core_co_labels"]
        )

    result = pd.DataFrame(selected_rows, columns=OUTPUT_COLUMNS)
    audit["selected_total"] = int(len(result))
    print("\nGoEmotions dominant-label filtering:")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return result, audit


def locate_emopars(root: Path) -> Path:
    candidates = [
        root / "data" / "raw" / "emopars.csv",
        root / "data" / "raw" / "emotion_fa" / "_sources" / "emopars" /
            "Persian-Emotion-Detection-main" / "dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        "Could not locate non-empty EmoPars CSV. Checked:\n" +
        "\n".join(str(path) for path in candidates)
    )


def deduplicate_rows(
    df: pd.DataFrame,
    *,
    description: str,
    split_priority: dict[str, int] | None = None,
    source_priority: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove empty texts, conflicting-label groups, and same-label duplicates."""
    if df.empty:
        stats = {"input_rows": 0, "empty_text_rows_removed": 0,
                 "conflicting_text_groups_removed": 0, "duplicate_rows_removed": 0,
                 "output_rows": 0}
        return df.copy(), stats

    work = df.copy().reset_index(drop=True)
    work["_text_key"] = work["text"].map(normalize_key)
    empty_mask = work["_text_key"].eq("")
    empty_removed = int(empty_mask.sum())
    work = work.loc[~empty_mask].copy()

    keep_indices: list[int] = []
    conflicts = 0
    duplicates_removed = 0
    for _, group in work.groupby("_text_key", sort=False):
        unique_labels = group["gold_label"].astype(str).unique()
        if len(unique_labels) > 1:
            conflicts += 1
            continue
        if len(group) > 1:
            duplicates_removed += len(group) - 1
            ranking = pd.DataFrame(index=group.index)
            if split_priority is not None and "split" in group.columns:
                ranking["split_rank"] = group["split"].map(split_priority).fillna(999)
            else:
                ranking["split_rank"] = 0
            if source_priority is not None and "source" in group.columns:
                ranking["source_rank"] = group["source"].map(source_priority).fillna(999)
            else:
                ranking["source_rank"] = 0
            ranking["row_rank"] = group.index
            keep_indices.append(int(ranking.sort_values(
                ["split_rank", "source_rank", "row_rank"]
            ).index[0]))
        else:
            keep_indices.append(int(group.index[0]))

    cleaned = work.loc[keep_indices].drop(columns="_text_key").reset_index(drop=True)
    stats = {
        "input_rows": int(len(df)),
        "empty_text_rows_removed": empty_removed,
        "conflicting_text_groups_removed": int(conflicts),
        "duplicate_rows_removed": int(duplicates_removed),
        "output_rows": int(len(cleaned)),
    }
    print(f"\n{description} deduplication:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return cleaned, stats


def stratified_emopars_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Create deterministic 80/10/10 splits, stratified by single dominant label."""
    work = df.copy().reset_index(drop=True)
    work["split"] = "train"
    rng = np.random.default_rng(seed)
    for label, group in work.groupby("gold_label", sort=True):
        indices = group.index.to_numpy()
        shuffled = rng.permutation(indices)
        n = len(shuffled)
        n_test = max(1, int(round(n * 0.10))) if n >= 3 else 0
        n_val = max(1, int(round(n * 0.10))) if n - n_test >= 2 else 0
        test_indices = shuffled[:n_test]
        val_indices = shuffled[n_test:n_test + n_val]
        work.loc[test_indices, "split"] = "test"
        work.loc[val_indices, "split"] = "validation"
    return work


def load_emopars(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = locate_emopars(root)
    df = read_csv_checked(path)
    text_col = find_column(df, "text", "sentence", "content")
    source_cols = {
        "anger": find_column(df, "Anger"),
        "fear": find_column(df, "Fear"),
        "joy": find_column(df, "Happiness", "Happy", "Joy"),
        "sadness": find_column(df, "Sadness"),
        "surprise": find_column(df, "Wonder", "Surprise"),
    }

    rows: list[dict[str, Any]] = []
    counts = Counter()
    for _, row in df.iterrows():
        text = "" if pd.isna(row[text_col]) else str(row[text_col])
        if not normalize_key(text):
            counts["empty_text"] += 1
            continue
        active: list[str] = []
        for label, column in source_cols.items():
            score = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(score) and float(score) >= EMOPARS_THRESHOLD:
                active.append(label)
        if len(active) == 0:
            counts["zero_core"] += 1
            continue
        if len(active) > 1:
            counts["multi_core"] += 1
            continue
        rows.append(build_row(
            text=text,
            label=active[0],
            source="emopars",
            language="fa",
            split="unassigned",
            original_labels=active[0],
        ))
        counts["selected_single_core"] += 1

    selected = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    deduped, dedup_stats = deduplicate_rows(selected, description="EmoPars single-label subset")
    split_df = stratified_emopars_split(deduped)
    audit = {
        "source_file": str(path),
        "input_rows": int(len(df)),
        "threshold": f">={EMOPARS_THRESHOLD} votes out of 5",
        "selected_single_core_before_dedup": int(counts["selected_single_core"]),
        "excluded_zero_core": int(counts["zero_core"]),
        "excluded_multi_core": int(counts["multi_core"]),
        "excluded_empty_text": int(counts["empty_text"]),
        "deduplication": dedup_stats,
        "split_counts": {str(k): int(v) for k, v in split_df["split"].value_counts().items()},
        "label_mapping": {"Happiness": "joy", "Wonder": "surprise", "Hatred": "excluded"},
    }
    print("\nEmoPars dominant-label filtering:")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return split_df, audit


def locate_armanemo(root: Path) -> tuple[Path, Path] | None:
    base = root / "data" / "raw" / "emotion_fa" / "_sources" / "armanemo" / \
        "arman-text-emotion-main" / "dataset"
    train_path, test_path = base / "train.tsv", base / "test.tsv"
    if train_path.exists() and test_path.exists():
        return train_path, test_path
    return None


def read_armanemo_tsv(path: Path) -> tuple[pd.DataFrame, int]:
    """ArmanEmo text may contain tabs; the label is parsed from the rightmost field."""
    rows: list[dict[str, str]] = []
    invalid = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line.strip() or "\t" not in line:
                if line.strip():
                    invalid += 1
                continue
            raw_text, raw_label = line.rsplit("\t", 1)
            label = raw_label.strip().strip('"').upper()
            if label not in ARMANEMO_ALL_LABELS:
                invalid += 1
                continue
            text_value = raw_text
            try:
                parsed = next(csv.reader([raw_text], delimiter="\t", quotechar='"', doublequote=True, strict=False))
                if len(parsed) == 1:
                    text_value = parsed[0]
            except Exception:
                pass
            rows.append({"text": text_value, "label": label})
    return pd.DataFrame(rows, columns=["text", "label"]), invalid


def load_armanemo(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = locate_armanemo(root)
    if paths is None:
        print("\nWARNING: ArmanEmo TSV files not found; skipping ArmanEmo.")
        return (pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame(columns=OUTPUT_COLUMNS),
                {"status": "skipped_missing_train_or_test_tsv"})

    train_path, test_path = paths
    raw_train, invalid_train = read_armanemo_tsv(train_path)
    raw_test, invalid_test = read_armanemo_tsv(test_path)

    def convert(frame: pd.DataFrame, split: str) -> tuple[pd.DataFrame, int, int]:
        rows = []
        excluded = 0
        empty = 0
        for _, row in frame.iterrows():
            original = str(row["label"]).strip().upper()
            mapped = ARMANEMO_TO_SHARED.get(original)
            if mapped is None:
                excluded += 1  # HATE and OTHER are intentionally outside the shared label set.
                continue
            text = str(row["text"])
            if not normalize_key(text):
                empty += 1
                continue
            rows.append(build_row(
                text=text,
                label=mapped,
                source="armanemo",
                language="fa",
                split=split,
                original_labels=original,
            ))
        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), excluded, empty

    train, train_excluded, train_empty = convert(raw_train, "train")
    external, test_excluded, test_empty = convert(raw_test, "external_test")

    # Keep ArmanEmo's own test split external and remove any same-text rows from its train split.
    external_keys = set(external["text"].map(normalize_key))
    overlap_mask = train["text"].map(normalize_key).isin(external_keys)
    overlap_removed = int(overlap_mask.sum())
    train = train.loc[~overlap_mask].copy()

    train_clean, train_dedup = deduplicate_rows(train, description="ArmanEmo mapped train")
    external_clean, external_dedup = deduplicate_rows(external, description="ArmanEmo external test")

    audit = {
        "train_source": str(train_path),
        "test_source": str(test_path),
        "raw_valid_train_rows": int(len(raw_train)),
        "raw_valid_test_rows": int(len(raw_test)),
        "invalid_train_rows": int(invalid_train),
        "invalid_test_rows": int(invalid_test),
        "excluded_train_HATE_OTHER": int(train_excluded),
        "excluded_test_HATE_OTHER": int(test_excluded),
        "empty_train_texts": int(train_empty),
        "empty_test_texts": int(test_empty),
        "train_rows_removed_due_to_train_test_text_overlap": overlap_removed,
        "train_deduplication": train_dedup,
        "external_test_deduplication": external_dedup,
        "mapped_train_rows": int(len(train_clean)),
        "mapped_external_test_rows": int(len(external_clean)),
    }
    print("\nArmanEmo dominant-label preparation:")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return train_clean, external_clean, audit


def resolve_main_duplicates(main: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    split_priority = {"test": 0, "validation": 1, "train": 2}
    source_priority = {"goemotions": 0, "emopars": 1, "armanemo": 2}
    return deduplicate_rows(
        main,
        description="Combined main emotion dataset",
        split_priority=split_priority,
        source_priority=source_priority,
    )


def summary_table(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total_rows": int(len(df)),
        "by_split": {},
        "by_language": {},
        "by_source": {},
        "class_counts": {label: int((df["gold_label"] == label).sum()) for label in LABELS},
    }
    for split, part in df.groupby("split", sort=True):
        result["by_split"][str(split)] = {
            "rows": int(len(part)),
            "by_language": {str(k): int(v) for k, v in part["language"].value_counts().items()},
            "by_source": {str(k): int(v) for k, v in part["source"].value_counts().items()},
            "class_counts": {label: int((part["gold_label"] == label).sum()) for label in LABELS},
        }
    for language, part in df.groupby("language", sort=True):
        result["by_language"][str(language)] = {
            "rows": int(len(part)),
            "by_source": {str(k): int(v) for k, v in part["source"].value_counts().items()},
            "class_counts": {label: int((part["gold_label"] == label).sum()) for label in LABELS},
        }
    for source, part in df.groupby("source", sort=True):
        result["by_source"][str(source)] = {
            "rows": int(len(part)),
            "by_split": {str(k): int(v) for k, v in part["split"].value_counts().items()},
            "class_counts": {label: int((part["gold_label"] == label).sum()) for label in LABELS},
        }
    return result


def overlap_groups(df: pd.DataFrame, column: str = "split") -> int:
    if df.empty:
        return 0
    temp = df.assign(_key=df["text"].map(normalize_key))
    temp = temp[temp["_key"] != ""]
    return int(temp.groupby("_key")[column].nunique().gt(1).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one-dominant-emotion datasets.")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent,
        help="Project root (default: folder containing this script).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / "data" / "processed" / "emotion_dominant_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DOMINANT EMOTION DATASET PREPARATION")
    print("=" * 72)
    print(f"Project root: {root}")
    print(f"Output directory: {output_dir}")
    print(f"Shared labels: {LABELS}")

    go, go_audit = load_goemotions(root)
    emo, emo_audit = load_emopars(root)
    arman_train, arman_external, arman_audit = load_armanemo(root)

    main_candidates = pd.concat([go, emo, arman_train], ignore_index=True)

    # Reserve all ArmanEmo external-test texts exclusively for external evaluation.
    external_keys = set(arman_external["text"].map(normalize_key)) if not arman_external.empty else set()
    if external_keys:
        overlap_mask = main_candidates["text"].map(normalize_key).isin(external_keys)
        external_overlap_removed = int(overlap_mask.sum())
        main_candidates = main_candidates.loc[~overlap_mask].copy()
    else:
        external_overlap_removed = 0

    main_clean, main_dedup = resolve_main_duplicates(main_candidates)
    external_clean, external_dedup = deduplicate_rows(
        arman_external, description="ArmanEmo reserved external test"
    )

    # Final leakage checks across internal splits and against the external test.
    cross_split = overlap_groups(main_clean, "split")
    external_key_set = set(external_clean["text"].map(normalize_key)) if not external_clean.empty else set()
    internal_external_overlap = sum(
        normalize_key(text) in external_key_set for text in main_clean["text"].tolist()
    )

    for split in ["train", "validation", "test"]:
        part = main_clean.loc[main_clean["split"] == split, OUTPUT_COLUMNS].copy()
        part = part.sample(frac=1, random_state=SEED).reset_index(drop=True)
        path = output_dir / f"{split}.csv"
        part.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\nWrote {len(part):,} rows -> {path}")

    external_path = output_dir / "armanemo_external_test.csv"
    external_clean[OUTPUT_COLUMNS].to_csv(external_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(external_clean):,} rows -> {external_path}")

    manifest = {
        "version": "emotion_dominant_v1",
        "seed": SEED,
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "task": "single-label dominant emotion classification",
        "target_column": "gold_label",
        "loss_recommendation": "CrossEntropyLoss",
        "goemotions_policy": (
            "Keep rows with exactly one of the five shared core emotions active. "
            "Ignore non-core labels for target selection; drop neutral combined with any other label. "
            "Preserve official train/validation/test assignments."
        ),
        "emopars_policy": (
            "Map Happiness->joy and Wonder->surprise; activate a label at >=3/5 votes; "
            "keep rows with exactly one active shared core label; drop zero-core and multi-core rows; "
            "stratified 80/10/10 split. Hatred is not mapped."
        ),
        "armanemo_policy": (
            "Map ANGRY/FEAR/HAPPY/SAD/SURPRISE to shared labels; exclude HATE and OTHER; "
            "use mapped train rows for training and reserve mapped test rows as external evaluation."
        ),
        "external_test_reserved_exclusively": True,
        "main_rows_removed_due_to_armanemo_external_test_overlap": external_overlap_removed,
        "main_deduplication": main_dedup,
        "armanemo_external_deduplication": external_dedup,
        "cross_split_duplicate_text_groups_after_cleaning": cross_split,
        "main_external_test_overlap_rows_after_cleaning": int(internal_external_overlap),
        "goemotions_audit": go_audit,
        "emopars_audit": emo_audit,
        "armanemo_audit": arman_audit,
        "main_dataset_summary": summary_table(main_clean),
        "armanemo_external_test_summary": summary_table(
            external_clean.assign(split="external_test") if not external_clean.empty else external_clean
        ),
        "important_limitations": [
            "A single dominant label is a task simplification; texts may genuinely contain multiple emotions.",
            "Wonder->surprise is an approximate cross-dataset mapping, not a guaranteed semantic identity.",
            "GoEmotions may retain non-core co-labels alongside one selected core label; inspect the audit count.",
            "ArmanEmo is a single-label source and its HATE/OTHER classes are excluded from the shared five-label task.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("FINAL DOMINANT-EMOTION DATASET SUMMARY")
    print("=" * 72)
    print(json.dumps(manifest["main_dataset_summary"], ensure_ascii=False, indent=2))
    print("\nArmanEmo external-test summary:")
    print(json.dumps(manifest["armanemo_external_test_summary"], ensure_ascii=False, indent=2))
    print("\nLeakage checks:")
    print(f"  Cross-split duplicate text groups: {cross_split}")
    print(f"  Main/external-test overlaps: {internal_external_overlap}")
    print(f"\nManifest: {manifest_path}")
    print("Done. Previous emotion_v1 and sentiment datasets were not modified.")

    if cross_split != 0 or internal_external_overlap != 0:
        raise RuntimeError(
            "Leakage check failed. Review manifest.json and duplicate audit before training."
        )


if __name__ == "__main__":
    main()
