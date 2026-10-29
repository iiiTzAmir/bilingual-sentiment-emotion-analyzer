from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Resolve the project root and load an optional local .env file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Set the Hugging Face cache directory before importing Transformers.
DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"
HF_HOME = Path(os.environ.get("HF_HOME", str(DEFAULT_HF_HOME))).expanduser()
os.environ.setdefault("HF_HOME", str(HF_HOME))

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Project layout expected:
# Emotion-Analyzer/
#   app/main.py
#   app/static/index.html
#   app/static/style.css
#   app/static/app.js
#   models/sentiment_multilingual_v1/   (for local mode)
#   models/emotion_dominant_v1/         (for local mode)
SENTIMENT_MODEL_DIR = PROJECT_ROOT / "models" / "sentiment_multilingual_v1"
EMOTION_MODEL_DIR = PROJECT_ROOT / "models" / "emotion_dominant_v1"

SENTIMENT_MODEL_ID = os.environ.get(
    "SENTIMENT_MODEL_ID", "iiTzAmir21/bilingual-sentiment-model"
)
EMOTION_MODEL_ID = os.environ.get(
    "EMOTION_MODEL_ID", "iiTzAmir21/bilingual-emotion-model"
)

# local: load model files from ./models (keeps current development workflow).
# hub: download the public model repositories from Hugging Face on startup.
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "local").strip().lower()
if MODEL_SOURCE not in {"local", "hub"}:
    raise ValueError("MODEL_SOURCE must be either 'local' or 'hub'.")

# HF_TOKEN is optional for public repositories. Never commit its value.
HF_TOKEN = os.environ.get("HF_TOKEN") or None
HF_REVISION = os.environ.get("HF_REVISION") or None

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_LENGTH = 128  # Transformer token limit
MAX_CHARACTERS = 128  # User-facing input limit
DEVICE = torch.device("cpu")  # Inference stays on CPU.


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARACTERS)


class TextClassifier:
    """Loads one sequence-classification model from local files or Hugging Face Hub."""

    def __init__(
        self,
        model_reference: str | Path,
        expected_labels: set[str],
        *,
        local_files_only: bool,
    ) -> None:
        reference = str(model_reference)
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "token": HF_TOKEN,
        }
        if HF_REVISION and not local_files_only:
            load_kwargs["revision"] = HF_REVISION

        self.tokenizer = AutoTokenizer.from_pretrained(reference, **load_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            reference,
            use_safetensors=True,
            **load_kwargs,
        )
        self.model.to(DEVICE)
        self.model.eval()

        raw_id2label = self.model.config.id2label
        self.id2label = {int(index): label for index, label in raw_id2label.items()}
        actual_labels = set(self.id2label.values())
        if actual_labels != expected_labels:
            raise ValueError(
                f"Unexpected labels in {reference}. "
                f"Expected {sorted(expected_labels)}, got {sorted(actual_labels)}"
            )

    def predict(self, text: str) -> dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        scores = {
            self.id2label[index]: round(float(probability), 6)
            for index, probability in enumerate(probabilities)
        }
        predicted_label = max(scores, key=scores.get)

        return {
            "label": predicted_label,
            "confidence": scores[predicted_label],
            "scores": scores,
        }


def detect_language(text: str) -> str:
    """Simple script-based detection for this Persian/English project."""
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text):
        return "fa"
    return "en"


def _model_reference(local_dir: Path, hub_id: str) -> tuple[str, bool]:
    """Return (model reference, local_files_only) for the configured source."""
    if MODEL_SOURCE == "local":
        if not local_dir.is_dir():
            raise FileNotFoundError(
                f"Local model directory not found: {local_dir}. "
                "Set MODEL_SOURCE=hub to load from Hugging Face instead."
            )
        return str(local_dir), True
    return hub_id, False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download/load each model exactly once when the API starts.
    sentiment_reference, sentiment_local_only = _model_reference(
        SENTIMENT_MODEL_DIR, SENTIMENT_MODEL_ID
    )
    emotion_reference, emotion_local_only = _model_reference(
        EMOTION_MODEL_DIR, EMOTION_MODEL_ID
    )

    app.state.sentiment_model = TextClassifier(
        sentiment_reference,
        {"negative", "neutral", "positive"},
        local_files_only=sentiment_local_only,
    )
    app.state.emotion_model = TextClassifier(
        emotion_reference,
        {"anger", "fear", "joy", "sadness", "surprise"},
        local_files_only=emotion_local_only,
    )
    yield


app = FastAPI(
    title="Bilingual Sentiment and Emotion Analyzer",
    description=(
        "Analyzes Persian and English text with two independent models: "
        "three-class sentiment and five-class dominant emotion."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

# The frontend and API share one origin, so no CORS configuration is needed.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": "cpu", "model_source": MODEL_SOURCE}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest, request: Request) -> dict[str, Any]:
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text must contain non-whitespace characters.",
        )
    if len(text) > MAX_CHARACTERS:
        raise HTTPException(
            status_code=422,
            detail=f"Text must be {MAX_CHARACTERS} characters or fewer.",
        )

    language = detect_language(text)
    sentiment = request.app.state.sentiment_model.predict(text)
    emotion = request.app.state.emotion_model.predict(text)

    return {
        "language": language,
        "sentiment": sentiment,
        "emotion": emotion,
    }
