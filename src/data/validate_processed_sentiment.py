
import csv
import json
import unicodedata
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(r"D:\NLP-Projects\Emotion-Analyzer")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS = ["sentiment_en", "sentiment_fa"]

VALID_LABELS = {"negative", "neutral", "positive"}


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ")
    text = text.casefold()
    return " ".join(text.split())


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def inspect_split(name, rows):
    labels = Counter(row["label"] for row in rows)
    normalized = [normalize_text(row["text"]) for row in rows]
    counts = Counter(normalized)

    duplicate_groups = sum(count > 1 for count in counts.values())
    empty_count = sum(not text for text in normalized)
    invalid_labels = sorted(set(labels) - VALID_LABELS)

    print(f"\n{name}")
    print("Rows:", len(rows))
    print("Label counts:", dict(sorted(labels.items())))
    print("Duplicate groups:", duplicate_groups)
    print("Empty texts:", empty_count)
    print("Invalid labels:", invalid_labels)

    return set(normalized), counts


def main():
    for dataset_name in DATASETS:
        folder = PROCESSED_DIR / dataset_name

        print("\n" + "=" * 65)
        print("VALIDATING:", dataset_name)
        print("=" * 65)

        splits = {}

        for split_name in [
            "train",
            "validation",
            "test",
            "test_official",
        ]:
            path = folder / f"{split_name}.csv"
            rows = read_csv(path)
            splits[split_name] = rows
            inspect_split(split_name, rows)

        normalized_sets = {
            name: {
                normalize_text(row["text"])
                for row in rows
                if normalize_text(row["text"])
            }
            for name, rows in splits.items()
        }

        print("\nCross-split overlap checks:")

        for first, second in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
            ("train", "test_official"),
            ("validation", "test_official"),
        ]:
            overlap = (
                normalized_sets[first] & normalized_sets[second]
            )
            print(f"{first} vs {second}: {len(overlap)}")

        # The clean test must be a subset of the official test.
        clean_test = normalized_sets["test"]
        official_test = normalized_sets["test_official"]

        missing_from_official = clean_test - official_test
        print(
            "Clean test texts absent from official test:",
            len(missing_from_official),
        )

        # Inspect saved cleaning report.
        report_path = folder / "cleaning_report.json"

        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        print(
            "\nSaved report clean sizes:",
            report["clean_split_sizes"],
        )

        print(
            "Actual CSV sizes:",
            {
                name: len(rows)
                for name, rows in splits.items()
            },
        )

        print("\nValidation finished.")


if __name__ == "__main__":
    main()
