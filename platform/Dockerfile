FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy everything needed for install
COPY pyproject.toml ./
COPY pwm_platform/ pwm_platform/
COPY scripts/ scripts/

# Install dependencies
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "pwm_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
