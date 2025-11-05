# Adaptive Momentum Strategy - Dockerfile
FROM python:3.11-slim

LABEL description="Adaptive Momentum Trading Bot with Dashboard Integration"
LABEL strategies="adaptive_momentum"
LABEL tier="competition"

WORKDIR /app

# Copy base infrastructure and strategy template files
COPY base-bot-template/ /app/base/
COPY your-strategy-template/ /app/

# Install dependencies once base requirements are available
RUN pip install --no-cache-dir -r /app/base/requirements.txt

RUN chmod +x /app/startup.py

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

EXPOSE 8080 3010

ENV BOT_EXCHANGE=paper
ENV BOT_STRATEGY=adaptive_momentum
ENV BOT_SYMBOL=BTC-USD
ENV BOT_STARTING_CASH=10000.0
ENV BOT_SLEEP=60

CMD ["python", "/app/startup.py"]
