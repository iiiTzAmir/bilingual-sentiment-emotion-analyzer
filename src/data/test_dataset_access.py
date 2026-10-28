
import os
import traceback
from pathlib import Path

# Keep Hugging Face cache on drive D
CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")

from datasets import load_dataset

DATASETS = [
    {
        "name": "TweetEval Sentiment",
        "repo": "cardiffnlp/tweet_eval",
        "config": "sentiment",
    },
    {
        "name": "SentiPers",
        "repo": "kforghani/sentipers",
        "config": None,
    },
]


def test_dataset(item):

    print("\n" + "=" * 60)
    print("Testing:", item["name"])
    print("=" * 60)

    try:
        if item["config"]:
            dataset = load_dataset(
                item["repo"],
                item["config"],
                cache_dir=str(CACHE_DIR),
            )
        else:
            dataset = load_dataset(
                item["repo"],
                cache_dir=str(CACHE_DIR),
            )

        print("\nSUCCESS!")
        print("Available splits:", list(dataset.keys()))

        for split_name, split_data in dataset.items():

            print(f"\nSplit: {split_name}")
            print("Samples:", len(split_data))
            print("Columns:", split_data.column_names)
            print("Features:", split_data.features)

            if len(split_data) > 0:
                print("First sample:", split_data[0])

    except Exception as error:

        print("\nFAILED!")
        print("Error type:", type(error).__name__)
        print("Error message:", str(error))

        print("\nFull traceback:")
        traceback.print_exc()


def main():

    print("DATASET CONNECTION DIAGNOSTIC")

    for item in DATASETS:
        test_dataset(item)

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
