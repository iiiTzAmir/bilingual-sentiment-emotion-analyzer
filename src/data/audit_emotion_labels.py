
import csv
from pathlib import Path
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "emotion_fa"
    / "_sources"
)

ARMAN_ROOT = (
    SOURCE_ROOT
    / "armanemo"
    / "arman-text-emotion-main"
    / "dataset"
)

EMOPARS_FILE = (
    SOURCE_ROOT
    / "emopars"
    / "Persian-Emotion-Detection-main"
    / "dataset.csv"
)


# --------------------------------------------------
# 1. Read ArmanEmo TSV without header or quote parsing
# --------------------------------------------------

VALID_LABELS = {
    "SAD",
    "HAPPY",
    "OTHER",
    "HATE",
    "ANGRY",
    "FEAR",
    "SURPRISE",
    "WONDER",
}


def inspect_arman_file(path):

    print("\n" + "=" * 65)
    print("ARMANEMO:", path.name)
    print("=" * 65)

    records = []
    invalid_rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            # Use the final tab as the label separator.
            # Quotes inside tweets remain ordinary characters.
            parts = line.rsplit("\t", 1)

            if len(parts) != 2:
                invalid_rows.append((line_number, line[:150]))
                continue

            text, label = parts

            text = text.strip()
            label = label.strip().upper()

            if label not in VALID_LABELS or not text:
                invalid_rows.append((line_number, line[:150]))
                continue

            records.append({
                "text": text,
                "label": label,
            })

    label_counts = Counter(
        row["label"] for row in records
    )

    print("Valid rows:", len(records))
    print("Invalid rows:", len(invalid_rows))
    print("Label distribution:")

    for label, count in sorted(label_counts.items()):
        percentage = count / len(records) * 100

        print(
            f"  {label}: {count} ({percentage:.2f}%)"
        )

    print("\nExamples:")

    for row in records[:3]:
        print("\nText:", row["text"][:200])
        print("Label:", row["label"])

    if invalid_rows:
        print("\nFirst invalid rows:")

        for line_number, content in invalid_rows[:5]:
            print(line_number, repr(content))

    return records


def inspect_arman():

    train_file = ARMAN_ROOT / "train.tsv"
    test_file = ARMAN_ROOT / "test.tsv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            "ArmanEmo train.tsv or test.tsv not found."
        )

    train = inspect_arman_file(train_file)
    test = inspect_arman_file(test_file)

    print("\nARMANEMO TOTAL")
    print("Train:", len(train))
    print("Test:", len(test))
    print("Total:", len(train) + len(test))


# --------------------------------------------------
# 2. Inspect EmoPars voting scores
# --------------------------------------------------

EMOTIONS = [
    "Anger",
    "Fear",
    "Happiness",
    "Hatred",
    "Sadness",
    "Wonder",
]


def inspect_emopars():

    print("\n" + "=" * 65)
    print("EMOPARS MULTI-LABEL AUDIT")
    print("=" * 65)

    df = pd.read_csv(
        EMOPARS_FILE,
        encoding="utf-8-sig",
    )

    print("Total samples:", len(df))
    print("Columns:", list(df.columns))

    scores = df[EMOTIONS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    print("\nMissing values:")
    print(scores.isna().sum().to_dict())

    print("\nInvalid values outside 0..5:")

    for column in EMOTIONS:

        invalid = (
            ~scores[column].between(0, 5)
        ) & scores[column].notna()

        print(column, int(invalid.sum()))

    # Explore both majority and stricter thresholds.
    for threshold in [3, 4]:

        print("\n" + "-" * 65)
        print(f"THRESHOLD: >= {threshold} votes")
        print("-" * 65)

        binary = scores.ge(threshold)

        label_counts = binary.sum(axis=0)

        for emotion, count in label_counts.items():

            percentage = count / len(df) * 100

            print(
                f"{emotion}: {int(count)} "
                f"({percentage:.2f}%)"
            )

        # Number of positive emotion labels per text.
        cardinality = binary.sum(axis=1)

        print("\nNumber of active labels per text:")

        distribution = (
            cardinality.value_counts()
            .sort_index()
        )

        for number, count in distribution.items():

            print(
                f"{number} labels: {int(count)} "
                f"({count / len(df) * 100:.2f}%)"
            )

        no_emotion = int((cardinality == 0).sum())
        multi_label = int((cardinality >= 2).sum())

        print("\nNo detected emotion:", no_emotion)
        print("Multi-label samples:", multi_label)

        print("\nExamples with multiple emotions:")

        shown = 0

        for index in binary.index:

            active = [
                emotion
                for emotion in EMOTIONS
                if binary.loc[index, emotion]
            ]

            if len(active) >= 2:

                print("\nText:", str(df.loc[index, "text"])[:180])
                print("Labels:", active)

                shown += 1

                if shown == 3:
                    break

        print("\nExamples without detected emotion:")

        shown = 0

        for index in binary.index:

            if cardinality.loc[index] == 0:

                print(
                    "\nText:",
                    str(df.loc[index, "text"])[:180],
                )

                shown += 1

                if shown == 3:
                    break


# --------------------------------------------------
# 3. Main
# --------------------------------------------------

def main():

    inspect_arman()
    inspect_emopars()

    print("\nEmotion dataset audit completed.")


if __name__ == "__main__":
    main()
