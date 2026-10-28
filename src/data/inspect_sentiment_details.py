
import os
import unicodedata
from pathlib import Path
from collections import Counter

PROJECT_CACHE = Path(r"D:\NLP-Projects\HF-Cache")

os.environ["HF_HOME"] = str(PROJECT_CACHE)
os.environ["HF_HUB_CACHE"] = str(PROJECT_CACHE / "hub")

from datasets import load_dataset


DATASETS = [
    {
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


def normalize_text(text):
    """Conservative normalization for duplicate inspection."""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ")
    text = text.casefold()

    return " ".join(text.split())


def inspect_dataset(spec):

    print("\n" + "=" * 65)
    print("DATASET:", spec["name"])
    print("=" * 65)

    if spec["config"]:
        dataset = load_dataset(
            spec["repo"],
            spec["config"],
            cache_dir=str(PROJECT_CACHE),
        )
    else:
        dataset = load_dataset(
            spec["repo"],
            cache_dir=str(PROJECT_CACHE),
        )

    split_texts = {}
    original_distribution = {}
    mapped_distribution = {}

    for split_name, split_data in dataset.items():

        original_counts = Counter()
        mapped_counts = Counter()
        normalized_texts = []
        empty_count = 0

        for example in split_data:

            text = example["text"] # type: ignore

            if not str(text).strip():
                empty_count += 1
                continue

            label_names = split_data.features["label"].names
            original_label = label_names[example["label"]] # type: ignore

            mapped_label = spec["label_map"][original_label]

            original_counts[original_label] += 1
            mapped_counts[mapped_label] += 1

            normalized_texts.append(normalize_text(text))

        text_set = set(normalized_texts)

        duplicate_excess = len(normalized_texts) - len(text_set)

        split_texts[split_name] = text_set

        original_distribution[split_name] = original_counts
        mapped_distribution[split_name] = mapped_counts

        print("\nSPLIT:", split_name)
        print("Total samples:", len(split_data))
        print("Empty texts:", empty_count)
        print("Duplicate occurrences:", duplicate_excess)

        print("\nOriginal label distribution:")

        for label, count in sorted(original_counts.items()):
            print(f"  {label}: {count}")

        print("\nMapped three-class distribution:")

        for label in ["negative", "neutral", "positive"]:
            count = mapped_counts[label]
            percentage = (
                count / sum(mapped_counts.values()) * 100
                if sum(mapped_counts.values()) else 0
            )

            print(
                f"  {label}: {count} ({percentage:.2f}%)"
            )

        print("\nSample examples:")

        shown = set()

        for example in split_data:

            original_label = (
                split_data.features["label"].names[
                    example["label"] # type: ignore
                ]
            )

            mapped_label = spec["label_map"][original_label]

            if mapped_label not in shown:

                print(
                    f"\nOriginal: {original_label}"
                )
                print(
                    f"Mapped: {mapped_label}"
                )
                print(
                    f"Text: {example['text'][:160]}" # type: ignore
                )

                shown.add(mapped_label)

            if len(shown) == 3:
                break

    print("\n" + "=" * 65)
    print("CROSS-SPLIT DUPLICATE CHECK")
    print("=" * 65)

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for first, second in pairs:

        if first not in split_texts or second not in split_texts:
            continue

        overlap = (
            split_texts[first] & split_texts[second]
        )

        print(
            f"{first} vs {second}: "
            f"{len(overlap)} overlapping texts"
        )

    print("\nDataset inspection finished.")


def main():

    for spec in DATASETS:

        try:
            inspect_dataset(spec)

        except Exception as error:
            print("\nERROR:", spec["name"])
            print(type(error).__name__, error)


if __name__ == "__main__":
    main()
