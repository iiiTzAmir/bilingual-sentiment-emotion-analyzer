# CPU-only container for the bilingual FastAPI inference service.
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MODEL_SOURCE=hub

WORKDIR /app

# Install CPU-only PyTorch. The official PyTorch CPU index avoids CUDA packages
# that are not needed on Cloud Run.
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       'torch>=2.6,<3.0'

COPY requirements-runtime.txt ./requirements-runtime.txt
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

# Copy only the API and UI. Model weights and datasets are downloaded/excluded.
COPY app ./app

# Run as a non-root user. Hugging Face cache is placed in writable /tmp.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /tmp/huggingface \
    && chown -R appuser:appuser /tmp/huggingface
USER appuser

EXPOSE 8080

# Cloud Run injects PORT. One worker ensures models are loaded only once.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
