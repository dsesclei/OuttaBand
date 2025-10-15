# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata ca-certificates bash \
 && rm -rf /var/lib/apt/lists/*

# install uv (fast resolver)
RUN pip install --no-cache-dir uv

WORKDIR /app

# cache deps
COPY pyproject.toml ./
# if you later add uv.lock, COPY it too for better caching
RUN uv pip compile pyproject.toml -o requirements.lock \
 && uv pip install --system --no-cache -r requirements.lock

# app layer
COPY . .

RUN groupadd --system appuser \
 && useradd --system --uid 10001 --gid appuser --create-home --shell /usr/sbin/nologin appuser \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ARG SERVICE_VERSION=dev
LABEL org.opencontainers.image.source="https://github.com/${GITHUB_REPOSITORY}" \
      org.opencontainers.image.version="${SERVICE_VERSION}" \
      org.opencontainers.image.title="OuttaBand" \
      org.opencontainers.image.description="async band watcher; demo ci/cd & sre"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import http.client; c=http.client.HTTPConnection('127.0.0.1',8000,timeout=5); c.request('GET','/healthz'); r=c.getresponse(); exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
