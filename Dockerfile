# syntax=docker/dockerfile:1
#
# Default image: API + UI + Groq client, TF-IDF embeddings and lexicon sentiment.
# Small (~600 MB) and starts in seconds. For FinBERT + MiniLM add the neural stack
# (adds PyTorch, ~2.5 GB):   docker build --build-arg WITH_NEURAL=1 -t finresearch:neural .
FROM python:3.11-slim

ARG WITH_NEURAL=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FINRESEARCH_LOG_JSON=1

WORKDIR /app

# dependencies first, so source edits do not invalidate this layer
COPY pyproject.toml README.md ./
RUN mkdir -p agents rag data/connectors eval api \
 && touch agents/__init__.py rag/__init__.py data/__init__.py data/connectors/__init__.py \
          eval/__init__.py api/__init__.py core.py llm.py pipeline.py config.py \
          sentiment_model.py logging_setup.py \
 && pip install ".[api,llm,ui]" \
 && if [ "$WITH_NEURAL" = "1" ]; then pip install ".[neural]"; fi \
 && pip uninstall -y finresearch-assistant

COPY . .

# run as an unprivileged user; /app/data/cache is the only writable path
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/data/cache && chown -R app:app /app/data/cache
USER app

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# one worker: the rate limiter and job store are in-process (see DEPLOY.md)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
