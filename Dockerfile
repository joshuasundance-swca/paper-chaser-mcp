FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv /opt/venv

COPY pyproject.toml README.md ./
COPY scholar_search_mcp ./scholar_search_mcp

RUN pip install --upgrade pip \
    && pip install . "uvicorn>=0.35.0"

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    SCHOLAR_SEARCH_TRANSPORT=streamable-http \
    SCHOLAR_SEARCH_HTTP_HOST=0.0.0.0 \
    SCHOLAR_SEARCH_HTTP_PORT=8080 \
    SCHOLAR_SEARCH_HTTP_PATH=/mcp

RUN groupadd --system app \
    && useradd --system --gid app --uid 10001 --create-home --home-dir /home/app app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz').read()"]

CMD ["python", "-m", "scholar_search_mcp.deployment_runner"]
