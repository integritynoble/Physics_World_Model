FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv
COPY pyproject.toml ./
COPY api/ api/
COPY workers/ workers/
COPY registry/ registry/
COPY bootstrap/ bootstrap/
COPY storage/ storage/
COPY ingestion/ ingestion/
COPY ui/ ui/
RUN uv pip install --system -e .
COPY migrations/ migrations/
ENV PYTHONPATH=/app
