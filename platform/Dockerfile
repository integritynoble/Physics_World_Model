FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only for CASSI DL model inference
RUN pip install --no-cache-dir "torch==2.3.0" \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "einops>=0.6.0"

# Copy and install pwm_core (simulation engine dependency)
COPY packages/pwm_core/ /tmp/pwm_core/
RUN cd /tmp/pwm_core && \
    sed -i 's|readme = "../../README.md"|readme = {text = "PWM Core", content-type = "text/plain"}|' pyproject.toml && \
    sed -i 's|license = {file = "../../LICENSE"}|license = {text = "MIT"}|' pyproject.toml && \
    pip install --no-cache-dir --no-deps . && \
    cd /app && rm -rf /tmp/pwm_core

# Copy everything needed for install
COPY platform/pyproject.toml ./
COPY platform/pwm_platform/ pwm_platform/
COPY platform/scripts/ scripts/

# Install dependencies
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "pwm_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
