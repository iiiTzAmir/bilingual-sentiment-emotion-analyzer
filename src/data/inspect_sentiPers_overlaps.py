
import os
import unicodedata
from pathlib import Path
from collections import Counter

CACHE_DIR = Path(r"D:\NLP-Projects\HF-Cache")

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")

from datasets import load_dataset


REPO = "kforghani/sentipers"

LABEL_MAP = {
    "very_negative": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very_positive": "positive",
}


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("\u200c", " ")
    text = text.casefold()

    return " ".join(text.split())


def display_text(text):
    """Make invisible and Arabic-specific characters visible."""

    return (
        str(text)
        .replace("\u200c", "<ZWNJ>")
        .replace("ي", "<ARABIC_YEH>")
        .replace("ك", "<ARABIC_KAF>")
    )


def collect_records(dataset):

    result = {}

    for split_name, split_data in dataset.items():

        result[split_name] = {}

        label_names = split_data.features["label"].names

        for example in split_data:

            raw_text = str(example["text"])
            normalized = normalize_text(raw_text)

            original_label = label_names[example["label"]]
            mapped_label = LABEL_MAP[original_label]

            result[split_name].setdefault(
                normalized,
                [],
            )

            result[split_name][normalized].append({
                "raw": raw_text,
                "original": original_label,
                "mapped": mapped_label,
            })

    return result


def print_group(normalized, first_records, second_records,
                first_name, second_name):

    print("\n" + "-" * 65)
    print("Normalized text:")
    print(normalized[:220])

    for split_name, records in [
        (first_name, first_records),
        (second_name, second_records),
    ]:

        print(f"\n{split_name}:")

        shown = set()

        for record in records:

            key = (
                record["raw"],
                record["original"],
            )

            if key in shown:
                continue

            shown.add(key)

            print("Original text:")
            print(display_text(record["raw"][:250]))

            print("Original label:", record["original"])
            print("Mapped label:", record["mapped"])


def main():

    print("Loading SentiPers...")

    dataset = load_dataset(
        REPO,
        cache_dir=str(CACHE_DIR),
    )

    records = collect_records(dataset)

    # --------------------------------------------------
    # 1. Internal duplicates
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("INTERNAL DUPLICATES")
    print("=" * 65)

    for split_name, split_records in records.items():

        duplicate_groups = [
            (text, items)
            for text, items in split_records.items()
            if len(items) > 1
        ]

        print(f"\nSPLIT: {split_name}")
        print("Duplicate groups:", len(duplicate_groups))

        for index, (text, items) in enumerate(
            duplicate_groups[:5],
            start=1,
        ):

            print(f"\nDuplicate group {index}:")
            print("Normalized:", text[:180])

            original_counts = Counter(
                item["original"] for item in items
            )

            mapped_counts = Counter(
                item["mapped"] for item in items
            )

            print("Original labels:", dict(original_counts))
            print("Mapped labels:", dict(mapped_counts))

            for item in items:
                print("\nRaw:", display_text(item["raw"][:200]))
                print("Label:", item["original"])

    # --------------------------------------------------
    # 2. Cross-split overlaps
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("CROSS-SPLIT OVERLAP DETAILS")
    print("=" * 65)

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for first, second in pairs:

        common = (
            set(records[first]) & set(records[second])
        )

        conflicting = []
        consistent = []

        exact_raw_count = 0

        for normalized in common:

            a = records[first][normalized]
            b = records[second][normalized]

            raw_a = {item["raw"] for item in a}
            raw_b = {item["raw"] for item in b}

            if raw_a & raw_b:
                exact_raw_count += 1

            labels_a = {item["mapped"] for item in a}
            labels_b = {item["mapped"] for item in b}

            if labels_a != labels_b:
                conflicting.append(normalized)
            else:
                consistent.append(normalized)

        print("\n" + "-" * 65)
        print(f"{first} vs {second}")
        print("Total overlaps:", len(common))
        print("Exact raw-text overlaps:", exact_raw_count)
        print("Mapped-label disagreements:", len(conflicting))
        print("Mapped-label agreement:", len(consistent))

        # Show conflicts first.
        examples = conflicting[:8]

        if len(examples) < 8:
            examples += consistent[:8 - len(examples)]

        for index, normalized in enumerate(examples, start=1):

            print(f"\nOVERLAP EXAMPLE {index}")

            print_group(
                normalized,
                records[first][normalized],
                records[second][normalized],
                first,
                second,
            )

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
