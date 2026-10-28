const MAX_CHARACTERS = 128;

const labels = {
  sentiment: {
    negative: { en: "Negative", fa: "منفی" },
    neutral: { en: "Neutral", fa: "خنثی" },
    positive: { en: "Positive", fa: "مثبت" },
  },
  emotion: {
    anger: { en: "Anger", fa: "خشم" },
    fear: { en: "Fear", fa: "ترس" },
    joy: { en: "Joy", fa: "شادی" },
    sadness: { en: "Sadness", fa: "غم" },
    surprise: { en: "Surprise", fa: "تعجب" },
  },
};

const input = document.getElementById("text-input");
const characterCount = document.getElementById("character-count");
const analyzeButton = document.getElementById("analyze-button");
const errorMessage = document.getElementById("error-message");
const resultsSection = document.getElementById("results-section");
const buttonLabel = analyzeButton.querySelector(".button-label");

function countCharacters(value) {
  return Array.from(value).length;
}

function updateCharacterCount() {
  const count = countCharacters(input.value);
  characterCount.textContent = `${count} / ${MAX_CHARACTERS}`;
  characterCount.classList.toggle("near-limit", count >= 110 && count < MAX_CHARACTERS);
  characterCount.classList.toggle("at-limit", count >= MAX_CHARACTERS);
}

function setError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function renderScores(targetId, scores, dictionary) {
  const container = document.getElementById(targetId);
  container.replaceChildren();

  Object.entries(scores).forEach(([key, value]) => {
    const translation = dictionary[key] || { en: key, fa: key };
    const item = document.createElement("div");
    item.className = "score-item";

    const label = document.createElement("div");
    label.className = "score-label";
    label.textContent = translation.fa;
    const english = document.createElement("small");
    english.textContent = translation.en;
    label.appendChild(english);

    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(100, Number(value) * 100))}%`;
    track.appendChild(fill);

    const score = document.createElement("div");
    score.className = "score-value";
    score.textContent = percent(value);

    item.append(label, track, score);
    container.appendChild(item);
  });
}

function renderPrediction(prefix, prediction, dictionary) {
  const translation = dictionary[prediction.label] || {
    en: prediction.label,
    fa: prediction.label,
  };
  const confidence = Math.max(0, Math.min(1, Number(prediction.confidence)));

  document.getElementById(`${prefix}-en`).textContent = translation.en;
  document.getElementById(`${prefix}-fa`).textContent = translation.fa;
  document.getElementById(`${prefix}-confidence`).textContent = percent(confidence);
  document.getElementById(`${prefix}-confidence-bar`).style.width = `${confidence * 100}%`;
  renderScores(`${prefix}-scores`, prediction.scores, dictionary);
}

function renderResults(data) {
  renderPrediction("sentiment", data.sentiment, labels.sentiment);
  renderPrediction("emotion", data.emotion, labels.emotion);

  const language = data.language === "fa"
    ? "فارسی · Persian"
    : "انگلیسی · English";
  document.getElementById("detected-language").textContent = `زبان تشخیص‌داده‌شده: ${language}`;

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyzeText() {
  const text = input.value.trim();
  const length = countCharacters(text);

  setError("");
  if (!text) {
    setError("لطفاً ابتدا یک متن وارد کن. · Please enter some text first.");
    input.focus();
    return;
  }
  if (length > MAX_CHARACTERS) {
    setError("طول متن نباید بیشتر از ۱۲۸ کاراکتر باشد. · Maximum length is 128 characters.");
    return;
  }

  analyzeButton.disabled = true;
  buttonLabel.textContent = "در حال تحلیل...";

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "در تحلیل متن مشکلی پیش آمد.";
      throw new Error(detail);
    }
    renderResults(data);
  } catch (error) {
    setError(`${error.message || "ارتباط با سرور برقرار نشد."} · لطفاً وضعیت اجرای API را بررسی کن.`);
  } finally {
    analyzeButton.disabled = false;
    buttonLabel.textContent = "تحلیل متن";
  }
}

input.addEventListener("input", updateCharacterCount);
analyzeButton.addEventListener("click", analyzeText);
input.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    analyzeText();
  }
});

updateCharacterCount();
