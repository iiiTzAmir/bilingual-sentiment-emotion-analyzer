
import os
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

PROJECT = Path(r"D:\NLP-Projects\Emotion-Analyzer")
CACHE = Path(r"D:\NLP-Projects\HF-Cache")

os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")

from datasets import load_dataset


# Five provisional shared emotions
CORE_EMOTIONS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

ARMA_LABEL_MAP = {
    "ANGRY": "anger",
    "FEAR": "fear",
    "HAPPY": "joy",
    "SAD": "sadness",
    "SURPRISE": "surprise",
}

EMOPARS_COLUMNS = {
    "Anger": "anger",
    "Fear": "fear",
    "Happiness": "joy",
    "Sadness": "sadness",
    "Wonder": "surprise",
}

EMOPARS_EXTRA = "Hatred"


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("\u200c", " ")
    text = text.casefold()
    return " ".join(text.split())


# --------------------------------------------------
# 1. GoEmotions audit
# --------------------------------------------------

def inspect_goemotions():

    print("\n" + "=" * 70)
    print("GOEMOTIONS ALIGNMENT")
    print("=" * 70)

    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified",
        cache_dir=str(CACHE),
    )

    shared = set(CORE_EMOTIONS)

    for split_name, split_data in dataset.items():

        label_names = (
            split_data.features["labels"].feature.names
        )

        counts = Counter()

        core_samples = 0
        neutral_only = 0
        non_core_only = 0
        mixed_neutral = 0

        for example in split_data:

            labels = {
                label_names[label]
                for label in example["labels"] # type: ignore
            }

            matched = labels & shared

            for emotion in matched:
                counts[emotion] += 1

            if matched:
                core_samples += 1

                if "neutral" in labels:
                    mixed_neutral += 1

            elif labels == {"neutral"}:
                neutral_only += 1

            else:
                non_core_only += 1

        print("\nSPLIT:", split_name)
        print("Total samples:", len(split_data))
        print("Samples with shared emotions:", core_samples)
        print("Neutral-only samples:", neutral_only)
        print("Only non-core labels:", non_core_only)
        print("Core + neutral combinations:", mixed_neutral)

        print("\nShared emotion counts:")

        for emotion in CORE_EMOTIONS:
            print(f"{emotion}: {counts[emotion]}")


# --------------------------------------------------
# 2. EmoPars duplicate and label audit
# --------------------------------------------------

def inspect_emopars():

    print("\n" + "=" * 70)
    print("EMOPARS ALIGNMENT")
    print("=" * 70)

    path = (
        PROJECT / "data" / "raw" / "emotion_fa"
        / "_sources" / "emopars"
        / "Persian-Emotion-Detection-main"
        / "dataset.csv"
    )

    df = pd.read_csv(path, encoding="utf-8-sig")

    label_columns = [
        "Anger",
        "Fear",
        "Happiness",
        "Hatred",
        "Sadness",
        "Wonder",
    ]

    scores = df[label_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    binary = scores.ge(3)

    print("Total samples:", len(df))

    print("\nEmotion counts at >= 3 votes:")

    for column in label_columns:
        print(
            f"{column}: {int(binary[column].sum())}"
        )

    core_columns = list(EMOPARS_COLUMNS.keys())

    core_binary = binary[core_columns]

    active_counts = core_binary.sum(axis=1)

    print("\nFive shared emotions:")
    print("Samples with at least one core emotion:",
          int((active_counts >= 1).sum()))
    print("Samples with no core emotion:",
          int((active_counts == 0).sum()))
    print("Samples with multiple core emotions:",
          int((active_counts >= 2).sum()))

    # Include Hatred separately.
    all_binary = binary[label_columns]
    all_active = all_binary.sum(axis=1)

    print("\nIncluding Persian-specific Hatred:")
    print("Zero-label samples:", int((all_active == 0).sum()))
    print("Multi-label samples:", int((all_active >= 2).sum()))

    # Check normalized duplicate groups.
    groups = defaultdict(list)

    for index, row in df.iterrows():

        normalized = normalize_text(row["text"])

        if normalized:
            groups[normalized].append(index)

    duplicate_groups = [
        indexes
        for indexes in groups.values()
        if len(indexes) > 1
    ]

    consistent = 0
    conflicts = 0

    for indexes in duplicate_groups:

        vectors = {
            tuple(int(binary.iloc[index][column])
                  for column in label_columns)
            for index in indexes
        }

        if len(vectors) == 1:
            consistent += 1
        else:
            conflicts += 1

    print("\nDuplicate audit:")
    print("Duplicate groups:", len(duplicate_groups))
    print("Same-label groups:", consistent)
    print("Different-label groups:", conflicts)


# --------------------------------------------------
# 3. ArmanEmo split audit
# --------------------------------------------------

def read_arman(path):

    records = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.rstrip("\r\n")
            parts = line.rsplit("\t", 1)

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid row at line {line_number}"
                )

            text, label = parts
            label = label.strip().upper()

            records.append({
                "text": text.strip(),
                "label": label,
                "normalized": normalize_text(text),
            })

    return records


def inspect_arman():

    print("\n" + "=" * 70)
    print("ARMANEMO ALIGNMENT")
    print("=" * 70)

    folder = (
        PROJECT / "data" / "raw" / "emotion_fa"
        / "_sources" / "armanemo"
        / "arman-text-emotion-main"
        / "dataset"
    )

    train = read_arman(folder / "train.tsv")
    test = read_arman(folder / "test.tsv")

    for name, rows in [("train", train), ("test", test)]:

        counts = Counter(row["label"] for row in rows)

        print(f"\n{name}: {len(rows)}")

        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")

        core_count = sum(
            counts[label] for label in ARMA_LABEL_MAP
        )

        print("Samples in five shared categories:", core_count)

    # Internal duplicate check
    for name, rows in [("train", train), ("test", test)]:

        groups = defaultdict(set)

        for row in rows:
            if row["normalized"]:
                groups[row["normalized"]].add(row["label"])

        duplicated = {
            text: labels
            for text, labels in groups.items()
            if sum(
                1 for row in rows
                if row["normalized"] == text
            ) > 1
        }

        conflicts = sum(
            len(labels) > 1
            for labels in duplicated.values()
        )

        print(f"\n{name} duplicate groups:", len(duplicated))
        print(f"{name} conflicting groups:", conflicts)

    train_texts = {
        row["normalized"]: row["label"]
        for row in train
        if row["normalized"]
    }

    test_texts = {
        row["normalized"]: row["label"]
        for row in test
        if row["normalized"]
    }

    overlap = set(train_texts) & set(test_texts)

    disagreements = sum(
        train_texts[text] != test_texts[text]
        for text in overlap
    )

    print("\nTrain/Test overlap:", len(overlap))
    print("Label disagreements:", disagreements)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    inspect_goemotions()
    inspect_emopars()
    inspect_arman()

    print("\nAlignment audit completed.")


if __name__ == "__main__":
    main()
