
import os
from pathlib import Path
from collections import Counter

# --------------------------------------------------
# 1. Configure Hugging Face cache on drive D
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()

CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")

# Import datasets after setting cache paths
from datasets import load_dataset


# --------------------------------------------------
# 2. Dataset configurations
# --------------------------------------------------

DATASETS = [
    {
        "name": "TweetEval - Sentiment (English)",
        "repo": "cardiffnlp/tweet_eval",
        "config": "sentiment",
        "text_column": "text",
        "label_column": "label",
    },
    {
        "name": "SentiPers (Persian)",
        "repo": "kforghani/sentipers",
        "config": None,
        "text_column": "text",
        "label_column": "label",
    },
    {
        "name": "GoEmotions - Simplified (English)",
        "repo": "google-research-datasets/go_emotions",
        "config": "simplified",
        "text_column": "text",
        "label_column": "labels",
    },
]

# Fallback mapping for SentiPers
SENTIPERS_LABELS = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive",
}


# --------------------------------------------------
# 3. Helper functions
# --------------------------------------------------

def get_label_names(feature):
    """Extract label names from Hugging Face features."""

    names = getattr(feature, "names", None)

    if names is not None:
        return names

    inner_feature = getattr(feature, "feature", None)

    if inner_feature is not None:
        return getattr(inner_feature, "names", None)

    return None


def convert_label(value, label_names, dataset_name):
    """Convert numerical label IDs into readable names."""

    custom_map = (
        SENTIPERS_LABELS
        if "SentiPers" in dataset_name
        else {}
    )

    if isinstance(value, (list, tuple)):
        return [
            convert_label(item, label_names, dataset_name)
            for item in value
        ]

    if isinstance(value, int):
        if label_names and 0 <= value < len(label_names):
            return label_names[value]

        if value in custom_map:
            return custom_map[value]

    return str(value)


# --------------------------------------------------
# 4. Inspect one dataset
# --------------------------------------------------

def inspect_dataset(spec):

    print("\n" + "=" * 65)
    print(f"DATASET: {spec['name']}")
    print("=" * 65)

    # Load dataset from Hugging Face
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

    print("\nRepository:", spec["repo"])
    print("Available splits:", list(dataset.keys()))

    label_column = spec["label_column"]
    text_column = spec["text_column"]

    # Global statistics
    all_labels = Counter()
    global_texts = {}

    for split_name, split_data in dataset.items():

        print("\n" + "-" * 45)
        print("SPLIT:", split_name)
        print("-" * 45)

        print("Number of samples:", len(split_data))
        print("Columns:", split_data.column_names)
        print("Features:", split_data.features)

        label_names = get_label_names(
            split_data.features[label_column]
        )

        if label_names:
            print("Label names:", label_names)

        empty_count = 0
        duplicate_count = 0

        split_texts = set()
        split_labels = Counter()
        text_lengths = []

        for example in split_data:

            text = str(example[text_column]).strip()

            if not text:
                empty_count += 1
                continue

            # Count duplicates within and across splits
            normalized_text = " ".join(
                text.casefold().split()
            )

            if normalized_text in split_texts:
                duplicate_count += 1

            if (
                normalized_text in global_texts
                and global_texts[normalized_text] != split_name
            ):
                pass

            global_texts.setdefault(
                normalized_text,
                split_name,
            )

            split_texts.add(normalized_text)

            text_lengths.append(len(text.split()))

            raw_label = example[label_column]

            if isinstance(raw_label, (list, tuple)):
                labels = raw_label
            else:
                labels = [raw_label]

            for label in labels:
                readable_label = convert_label(
                    label,
                    label_names,
                    spec["name"],
                )

                split_labels[readable_label] += 1
                all_labels[readable_label] += 1

        print("Empty texts:", empty_count)
        print("Duplicate texts within split:", duplicate_count)

        if text_lengths:
            average_length = sum(text_lengths) / len(text_lengths)
            print(f"Average words per text: {average_length:.2f}")

        print("\nLabel distribution:")

        for label, count in split_labels.most_common():
            print(f"  {label}: {count}")

        print("\nSample examples:")

        shown = 0

        for example in split_data:

            text = str(example[text_column]).strip()

            if not text:
                continue

            readable_label = convert_label(
                example[label_column],
                label_names,
                spec["name"],
            )

            print(f"\nText: {text[:180]}")
            print(f"Label: {readable_label}")

            shown += 1

            if shown == 2:
                break

    print("\n" + "=" * 65)
    print("TOTAL LABEL DISTRIBUTION:", spec["name"])
    print("=" * 65)

    for label, count in all_labels.most_common():
        print(f"{label}: {count}")

    print("\nInspection completed successfully.")


# --------------------------------------------------
# 5. Main
# --------------------------------------------------

def main():

    for spec in DATASETS:

        try:
            inspect_dataset(spec)

        except Exception as error:
            print("\nERROR while inspecting:", spec["name"])
            print(type(error).__name__, ":", error)

    print("\nAll dataset inspections finished.")


if __name__ == "__main__":
    main()
