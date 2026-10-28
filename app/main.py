from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
#   models/sentiment_multilingual_v1/
#   models/emotion_dominant_v1/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_MODEL_DIR = PROJECT_ROOT / "models" / "sentiment_multilingual_v1"
EMOTION_MODEL_DIR = PROJECT_ROOT / "models" / "emotion_dominant_v1"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_LENGTH = 128  # Transformer token limit
MAX_CHARACTERS = 128  # User-facing input limit
DEVICE = torch.device("cpu")  # Inference stays on CPU.


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARACTERS)


class TextClassifier:
    """Loads one local Hugging Face sequence-classification model."""

    def __init__(self, model_dir: Path, expected_labels: set[str]) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self.model.to(DEVICE)
        self.model.eval()

        raw_id2label = self.model.config.id2label
        self.id2label = {int(index): label for index, label in raw_id2label.items()}
        actual_labels = set(self.id2label.values())
        if actual_labels != expected_labels:
            raise ValueError(
                f"Unexpected labels in {model_dir.name}. "
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load each model exactly once when the API starts.
    app.state.sentiment_model = TextClassifier(
        SENTIMENT_MODEL_DIR,
        {"negative", "neutral", "positive"},
    )
    app.state.emotion_model = TextClassifier(
        EMOTION_MODEL_DIR,
        {"anger", "fear", "joy", "sadness", "surprise"},
    )
    yield


app = FastAPI(
    title="Bilingual Sentiment and Emotion Analyzer",
    description=(
        "Analyzes Persian and English text with two independent models: "
        "three-class sentiment and five-class dominant emotion."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# The frontend and API share one origin, so no CORS configuration is needed.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": "cpu"}


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
