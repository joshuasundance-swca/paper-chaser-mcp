# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv /opt/venv

COPY pyproject.toml README.md LICENSE ./
COPY scholar_search_mcp ./scholar_search_mcp

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install ".[ai,ai-faiss]"

FROM python:${PYTHON_VERSION}-slim AS runtime

ARG UID=10001
ARG GID=10001
ARG VERSION=dev
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    SCHOLAR_SEARCH_HTTP_HOST=0.0.0.0 \
    SCHOLAR_SEARCH_HTTP_PORT=8080 \
    SCHOLAR_SEARCH_HTTP_PATH=/mcp

LABEL org.opencontainers.image.title="Scholar Search MCP" \
      org.opencontainers.image.description="Academic paper discovery MCP server with stdio-first local packaging and optional HTTP deployment wrapper." \
      org.opencontainers.image.documentation="https://github.com/joshuasundance-swca/scholar-search-mcp#readme" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/joshuasundance-swca/scholar-search-mcp" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}" \
      io.modelcontextprotocol.server.name="io.github.joshuasundance-swca/scholar-search-mcp"

RUN groupadd --gid "${GID}" app \
    && useradd --uid "${UID}" --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /workspace \
    && chown -R app:app /workspace /home/app

WORKDIR /workspace

COPY --from=builder /opt/venv /opt/venv

USER app

EXPOSE 8080

ENTRYPOINT ["scholar-search-mcp"]

CMD []
