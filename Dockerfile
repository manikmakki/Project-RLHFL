FROM python:3.13-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU-only) separately first for layer caching
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY services/ ./
COPY scripts/ ./scripts/
COPY start.sh ./

RUN chmod +x start.sh

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["./start.sh"]
