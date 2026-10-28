
import os
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")

from datasets import load_dataset


DATASETS = [
    {
        "name": "TweetEval",
        "repo": "cardiffnlp/tweet_eval",
        "config": "sentiment",
        "mapping": {
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
        },
    },
    {
        "name": "SentiPers",
        "repo": "kforghani/sentipers",
        "config": None,
        "mapping": {
            "very_negative": "negative",
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
            "very_positive": "positive",
        },
    },
]


def normalize_text(text):
    """Normalize text only for duplicate detection."""

    text = unicodedata.normalize("NFKC", str(text))

    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")

    text = text.replace("\u200c", " ")
    text = text.casefold()

    return " ".join(text.split())


def load_data(spec):

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

    return dataset


def audit_dataset(spec):

    print("\n" + "=" * 70)
    print("DUPLICATE AUDIT:", spec["name"])
    print("=" * 70)

    dataset = load_data(spec)

    records = {}

    # Read every split and group normalized texts.
    for split_name, split_data in dataset.items():

        records[split_name] = {}

        label_names = split_data.features["label"].names

        for example in split_data:

            raw_text = str(example["text"]) # type: ignore
            normalized = normalize_text(raw_text)

            if normalized not in records[split_name]:

                records[split_name][normalized] = {
                    "count": 0,
                    "original_labels": Counter(),
                    "mapped_labels": Counter(),
                    "raw_texts": set(),
                    "example": raw_text,
                }

            item = records[split_name][normalized]

            original_label = label_names[
                example["label"] # type: ignore
            ]

            mapped_label = spec["mapping"][original_label]

            item["count"] += 1
            item["original_labels"][original_label] += 1
            item["mapped_labels"][mapped_label] += 1
            item["raw_texts"].add(raw_text)

    # --------------------------------------------------
    # 1. Duplicate analysis inside each split
    # --------------------------------------------------

    for split_name, split_records in records.items():

        duplicate_groups = [
            item
            for item in split_records.values()
            if item["count"] > 1
        ]

        duplicate_occurrences = sum(
            item["count"] - 1
            for item in duplicate_groups
        )

        original_conflicts = [
            item
            for item in duplicate_groups
            if len(item["original_labels"]) > 1
        ]

        mapped_conflicts = [
            item
            for item in duplicate_groups
            if len(item["mapped_labels"]) > 1
        ]

        print("\nSPLIT:", split_name)
        print("Unique normalized texts:", len(split_records))
        print("Duplicate groups:", len(duplicate_groups))
        print("Extra duplicate occurrences:", duplicate_occurrences)

        print(
            "Original-label conflict groups:",
            len(original_conflicts),
        )

        print(
            "Mapped-label conflict groups:",
            len(mapped_conflicts),
        )

        for index, item in enumerate(
            original_conflicts[:3],
            start=1,
        ):

            print(f"\nConflict example {index}:")
            print("Text:", item["example"][:150])
            print("Original labels:", dict(item["original_labels"]))
            print("Mapped labels:", dict(item["mapped_labels"]))

    # --------------------------------------------------
    # 2. Cross-split overlap analysis
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("CROSS-SPLIT ANALYSIS")
    print("-" * 70)

    for first, second in combinations(records.keys(), 2):

        first_data = records[first]
        second_data = records[second]

        common_texts = set(first_data) & set(second_data)

        raw_exact = 0
        original_conflicts = 0
        mapped_conflicts = 0

        conflict_examples = []

        for text in common_texts:

            a = first_data[text]
            b = second_data[text]

            # Check exact raw text overlap.
            if a["raw_texts"] & b["raw_texts"]:
                raw_exact += 1

            labels_a = set(a["original_labels"])
            labels_b = set(b["original_labels"])

            mapped_a = set(a["mapped_labels"])
            mapped_b = set(b["mapped_labels"])

            if labels_a != labels_b:
                original_conflicts += 1

            if mapped_a != mapped_b:
                mapped_conflicts += 1

            if (
                labels_a != labels_b
                or mapped_a != mapped_b
            ):
                conflict_examples.append(
                    (text, a, b)
                )

        print(f"\n{first} vs {second}")

        print("Normalized overlapping texts:", len(common_texts))
        print("Exact raw-text overlaps:", raw_exact)

        print("Original-label disagreements:", original_conflicts)
        print("Mapped-label disagreements:", mapped_conflicts)

        for index, (text, a, b) in enumerate(
            conflict_examples[:5],
            start=1,
        ):

            print(f"\nConflict example {index}:")
            print("Text:", a["example"][:150])

            print(
                f"{first} original labels:",
                dict(a["original_labels"]),
            )

            print(
                f"{second} original labels:",
                dict(b["original_labels"]),
            )

            print(
                f"{first} mapped labels:",
                dict(a["mapped_labels"]),
            )

            print(
                f"{second} mapped labels:",
                dict(b["mapped_labels"]),
            )

    print("\nAudit completed:", spec["name"])


def main():

    for spec in DATASETS:

        try:
            audit_dataset(spec)

        except Exception as error:
            print("\nERROR:", spec["name"])
            print(type(error).__name__, error)


if __name__ == "__main__":
    main()
