FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser \
    && useradd --system --uid 10001 --gid appuser --create-home --shell /usr/sbin/nologin appuser

RUN mkdir -p /app /data \
    && chown appuser:appuser /app /data

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import http.client; conn = http.client.HTTPConnection('127.0.0.1', 8000, timeout=5); conn.request('GET', '/healthz'); resp = conn.getresponse(); exit(0 if resp.status == 200 else 1)"

USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
