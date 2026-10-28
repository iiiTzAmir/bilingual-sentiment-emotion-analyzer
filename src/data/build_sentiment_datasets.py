
import os
import csv
import json
import unicodedata

from pathlib import Path
from collections import defaultdict


# --------------------------------------------------
# 1. Paths and Hugging Face cache
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")

from datasets import load_dataset


# --------------------------------------------------
# 2. Dataset definitions
# --------------------------------------------------

DATASETS = [
    {
        "key": "sentiment_en",
        "name": "TweetEval",
        "repo": "cardiffnlp/tweet_eval",
        "config": "sentiment",
        "label_map": {
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
        },
    },
    {
        "key": "sentiment_fa",
        "name": "SentiPers",
        "repo": "kforghani/sentipers",
        "config": None,
        "label_map": {
            "very_negative": "negative",
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
            "very_positive": "positive",
        },
    },
]


# --------------------------------------------------
# 3. Text normalization for duplicate detection
# --------------------------------------------------

def normalize_text(text):
    """
    Conservative normalization used ONLY for duplicate detection.

    The original text is preserved for model training.
    """

    text = unicodedata.normalize("NFKC", str(text))

    # Normalize common Persian/Arabic character variants.
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")

    # Treat ZWNJ and ordinary spaces consistently.
    text = text.replace("\u200c", " ")

    # Normalize whitespace and letter case.
    text = text.casefold()

    return " ".join(text.split())


# --------------------------------------------------
# 4. Load source dataset
# --------------------------------------------------

def load_source_dataset(spec):

    print("\nLoading:", spec["name"])

    if spec["config"]:
        dataset = load_dataset(
            spec["repo"],
            spec["config"],
            cache_dir=str(CACHE_DIR),
        )
    else:
        dataset = load_dataset(
            spec["repo"],
            cache_dir=str(CACHE_DIR),
        )

    all_splits = {}

    for split_name, split_data in dataset.items():

        label_names = split_data.features["label"].names
        rows = []

        for index, example in enumerate(split_data):

            source_label_id = int(example["label"]) # type: ignore
            source_label = label_names[source_label_id]

            if source_label not in spec["label_map"]:
                raise ValueError(
                    f"Unknown label {source_label!r} "
                    f"in {spec['name']}"
                )

            text = str(example["text"]) # type: ignore

            rows.append({
                "text": text,
                "source_label_id": source_label_id,
                "source_label": source_label,
                "label": spec["label_map"][source_label],
                "source_index": index,
                "_norm": normalize_text(text),
            })

        all_splits[split_name] = rows

    return all_splits


# --------------------------------------------------
# 5. CSV helpers
# --------------------------------------------------

RAW_COLUMNS = [
    "text",
    "source_label_id",
    "source_label",
    "source_index",
]

PROCESSED_COLUMNS = [
    "text",
    "label",
    "source_label",
    "source_label_id",
    "source_index",
]


def write_csv(path, rows, columns):

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_raw_splits(dataset_key, splits):

    destination = RAW_DIR / dataset_key

    for split_name, rows in splits.items():

        write_csv(
            destination / f"{split_name}.csv",
            rows,
            RAW_COLUMNS,
        )


# --------------------------------------------------
# 6. Remove overlaps across splits
# --------------------------------------------------

def remove_cross_split_overlaps(splits):

    # Test is preserved as the reference evaluation split.
    test_norms = {
        row["_norm"]
        for row in splits["test"]
        if row["_norm"]
    }

    validation_norms = {
        row["_norm"]
        for row in splits["validation"]
        if row["_norm"]
    }

    original_counts = {
        name: len(rows)
        for name, rows in splits.items()
    }

    # Remove training texts appearing in validation or test.
    clean_train = [
        row
        for row in splits["train"]
        if row["_norm"]
        and row["_norm"] not in validation_norms
        and row["_norm"] not in test_norms
    ]

    # Remove validation texts appearing in test.
    clean_validation = [
        row
        for row in splits["validation"]
        if row["_norm"]
        and row["_norm"] not in test_norms
    ]

    # Keep all original test rows at this stage.
    clean_test = list(splits["test"])

    cleaned = {
        "train": clean_train,
        "validation": clean_validation,
        "test": clean_test,
    }

    removed = {
        "train_cross_split": (
            original_counts["train"] - len(clean_train)
        ),
        "validation_test_overlap": (
            original_counts["validation"] - len(clean_validation)
        ),
    }

    return cleaned, removed


# --------------------------------------------------
# 7. Remove duplicate and ambiguous groups
# --------------------------------------------------

def clean_within_split(rows):

    groups = defaultdict(list)

    for row in rows:

        if row["_norm"]:
            groups[row["_norm"]].append(row)

    clean_rows = []

    duplicate_extra_rows = 0
    conflicting_groups = 0
    conflicting_rows = 0

    for normalized_text, group in groups.items():

        mapped_labels = {
            row["label"]
            for row in group
        }

        # If the same normalized text has different target
        # labels, exclude the entire group from the clean split.
        if len(mapped_labels) > 1:

            conflicting_groups += 1
            conflicting_rows += len(group)
            continue

        # Keep one copy for groups with a consistent target label.
        clean_rows.append(group[0])

        duplicate_extra_rows += len(group) - 1

    stats = {
        "input_rows": len(rows),
        "output_rows": len(clean_rows),
        "duplicate_extra_rows_removed": duplicate_extra_rows,
        "conflicting_groups_removed": conflicting_groups,
        "conflicting_rows_removed": conflicting_rows,
        "empty_text_rows_removed": sum(
            1 for row in rows if not row["_norm"]
        ),
    }

    return clean_rows, stats


# --------------------------------------------------
# 8. Verify no cross-split overlap remains
# --------------------------------------------------

def verify_no_cross_split_overlap(splits):

    norm_sets = {
        name: {row["_norm"] for row in rows}
        for name, rows in splits.items()
    }

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    results = {}

    for first, second in pairs:

        overlap = norm_sets[first] & norm_sets[second]

        results[f"{first}_vs_{second}"] = len(overlap)

        if overlap:
            raise RuntimeError(
                f"Cross-split overlap remains: "
                f"{first} vs {second}: {len(overlap)}"
            )

    return results


# --------------------------------------------------
# 9. Build clean dataset
# --------------------------------------------------

def build_dataset(spec):

    original_splits = load_source_dataset(spec)

    # Save source exports before filtering.
    save_raw_splits(spec["key"], original_splits)

    # Remove potential leakage across splits.
    split_rows, cross_split_stats = (
        remove_cross_split_overlaps(original_splits)
    )

    # Clean duplicates within each split.
    clean_splits = {}
    split_stats = {}

    for split_name, rows in split_rows.items():

        clean_rows, stats = clean_within_split(rows)

        clean_splits[split_name] = clean_rows
        split_stats[split_name] = stats

    # Keep an unfiltered, target-label-mapped test export.
    official_test = list(original_splits["test"])

    # Verify the cleaned sets are mutually disjoint.
    overlaps = verify_no_cross_split_overlap(clean_splits)

    # Write processed files.
    destination = PROCESSED_DIR / spec["key"]

    for split_name, rows in clean_splits.items():

        write_csv(
            destination / f"{split_name}.csv",
            rows,
            PROCESSED_COLUMNS,
        )

    # Save the complete original test split with mapped labels.
    write_csv(
        destination / "test_official.csv",
        official_test,
        PROCESSED_COLUMNS,
    )

    report = {
        "dataset": spec["name"],
        "repository": spec["repo"],
        "label_mapping": spec["label_map"],
        "original_split_sizes": {
            name: len(rows)
            for name, rows in original_splits.items()
        },
        "cross_split_rows_removed": cross_split_stats,
        "cleaning_stats": split_stats,
        "remaining_cross_split_overlaps": overlaps,
        "clean_split_sizes": {
            name: len(rows)
            for name, rows in clean_splits.items()
        },
        "official_test_size": len(official_test),
    }

    report_path = destination / "cleaning_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("CLEANING SUMMARY:", spec["name"])
    print("=" * 65)

    print("Original sizes:", report["original_split_sizes"])
    print("Cross-split rows removed:", cross_split_stats)
    print("Clean sizes:", report["clean_split_sizes"])
    print("Official test size:", len(official_test))

    for split_name, stats in split_stats.items():
        print(f"\n{split_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\nRemaining overlaps:", overlaps)
    print("Report saved to:", report_path)


def main():

    for spec in DATASETS:
        build_dataset(spec)

    print("\nAll sentiment datasets have been processed.")


if __name__ == "__main__":
    main()
