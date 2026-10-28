
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(r"D:\NLP-Projects\Emotion-Analyzer")

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "emotion_fa"
    / "_sources"
)

SOURCE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Official source repositories
# --------------------------------------------------

REPOSITORIES = [
    {
        "name": "ArmanEmo",
        "url": (
            "https://github.com/"
            "Arman-Rayan-Sharif/"
            "arman-text-emotion/"
            "archive/refs/heads/main.zip"
        ),
        "archive": SOURCE_DIR / "armanemo_source.zip",
        "extract_dir": SOURCE_DIR / "armanemo",
    },
    {
        "name": "EmoPars",
        "url": (
            "https://github.com/"
            "nazaninsbr/"
            "Persian-Emotion-Detection/"
            "archive/refs/heads/main.zip"
        ),
        "archive": SOURCE_DIR / "emopars_source.zip",
        "extract_dir": SOURCE_DIR / "emopars",
    },
]


# --------------------------------------------------
# 3. Download with retries
# --------------------------------------------------

def download_archive(item):

    archive_path = item["archive"]

    if archive_path.exists() and zipfile.is_zipfile(archive_path):
        print("Archive already exists:", archive_path.name)
        return

    print("\nDownloading:", item["name"])
    print("URL:", item["url"])

    temporary_path = archive_path.with_suffix(".part")

    headers = {
        "User-Agent": "Emotion-Analyzer-Dataset-Inspector/1.0"
    }

    last_error = None

    for attempt in range(1, 4):

        try:
            with requests.get(
                item["url"],
                headers=headers,
                stream=True,
                timeout=(20, 120),
            ) as response:

                response.raise_for_status()

                with temporary_path.open("wb") as file:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 128
                    ):
                        if chunk:
                            file.write(chunk)

            if not zipfile.is_zipfile(temporary_path):
                raise ValueError(
                    "The downloaded file is not a valid ZIP archive."
                )

            temporary_path.replace(archive_path)

            print("Downloaded successfully.")
            return

        except Exception as error:

            last_error = error

            print(
                f"Attempt {attempt}/3 failed:",
                type(error).__name__,
                str(error),
            )

            temporary_path.unlink(missing_ok=True)

            if attempt < 3:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Could not download {item['name']}: {last_error}"
    )


# --------------------------------------------------
# 4. Safely extract ZIP files
# --------------------------------------------------

def extract_archive(item):

    destination = item["extract_dir"]

    if destination.exists():
        print("\nExtraction directory already exists:", destination)
        return

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(item["archive"], "r") as archive:

        root = destination.resolve()

        # Prevent ZIP path traversal.
        for member in archive.infolist():

            target = (destination / member.filename).resolve()

            if not target.is_relative_to(root):
                raise ValueError(
                    f"Unsafe ZIP entry detected: {member.filename}"
                )

        archive.extractall(destination)

    print("Extracted to:", destination)


# --------------------------------------------------
# 5. Inspect files and tabular data
# --------------------------------------------------

TABULAR_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".parquet",
}


def read_table(path):

    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    # Let pandas infer the delimiter, supporting CSV and TSV.
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        on_bad_lines="warn",
    )


def inspect_repository(item):

    print("\n" + "=" * 75)
    print("REPOSITORY:", item["name"])
    print("=" * 75)

    extract_archive(item)

    root = item["extract_dir"]

    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
    )

    print("\nAll repository files:")

    for path in files:
        relative = path.relative_to(root)
        size_kb = path.stat().st_size / 1024

        print(f"  {relative} ({size_kb:.1f} KB)")

    candidates = [
        path for path in files
        if path.suffix.lower() in TABULAR_EXTENSIONS
    ]

    if not candidates:
        print("\nNo supported tabular files found.")
        return

    print("\n" + "-" * 75)
    print("TABULAR FILE INSPECTION")
    print("-" * 75)

    for path in candidates:

        print("\nFILE:", path.relative_to(root))

        try:
            df = read_table(path)

            print("Rows:", len(df))
            print("Columns:", len(df.columns))
            print("Column names:", list(df.columns))

            print("\nFirst 3 rows:")

            print(
                df.head(3).to_string(
                    index=False,
                    max_colwidth=90,
                )
            )

            # Print distributions for likely label columns.
            for column in df.columns:

                name = str(column).strip().lower()

                if any(
                    keyword in name
                    for keyword in [
                        "label",
                        "emotion",
                        "sentiment",
                        "class",
                        "anger",
                        "fear",
                        "happiness",
                        "sadness",
                        "hatred",
                        "wonder",
                    ]
                ):

                    unique_count = df[column].nunique(
                        dropna=False
                    )

                    if unique_count <= 20:
                        print(
                            f"\nValue counts for {column!r}:"
                        )

                        print(
                            df[column]
                            .value_counts(dropna=False)
                            .head(20)
                            .to_string()
                        )

        except Exception as error:

            print(
                "Could not parse this file:",
                type(error).__name__,
                str(error),
            )


# --------------------------------------------------
# 6. Main
# --------------------------------------------------

def main():

    for item in REPOSITORIES:

        try:
            download_archive(item)
            inspect_repository(item)

        except Exception as error:

            print("\nFAILED:", item["name"])
            print(type(error).__name__, str(error))

    print("\nInspection finished.")


if __name__ == "__main__":
    main()
