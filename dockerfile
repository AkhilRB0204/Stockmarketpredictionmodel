FROM --platform=$BUILDPLATFORM python:3.11-slim

# System deps for matplotlib fonts + yfinance SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY data.py model.py validation.py app.py ./

# Output volume — model.pkl and PNGs land here
RUN mkdir -p /app/output
VOLUME ["/app/output"]

# Flask on 5000
EXPOSE 5000

# Health check — hits the /health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

ENV TICKER=AAPL \
    RETRAIN_EVERY=30 \
    MIN_RETRAIN_INTERVAL=10 \
    POLL_INTERVAL=60

CMD ["python", "app.py"]