# DCA Bot Template - Dockerfile
FROM python:3.11-slim

LABEL description="DCA Trading Bot with Dashboard Integration"
LABEL strategies="dca,advanced_dca"
LABEL tier="basic,enterprise"

# Set working directory
WORKDIR /app

# Copy base bot infrastructure
COPY base-bot-template/ /app/base/

# Copy DCA-specific files
COPY dca-bot-template/ /app/

# Install dependencies
RUN pip install --no-cache-dir -r /app/base/requirements.txt

# Make startup script executable
RUN chmod +x /app/startup.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Expose ports
EXPOSE 8080 3010

# Default environment
ENV BOT_EXCHANGE=paper
ENV BOT_STRATEGY=dca
ENV BOT_SYMBOL=BTC-USD
ENV BOT_STARTING_CASH=1000.0
ENV BOT_SLEEP=60

# Start the DCA bot
CMD ["python", "/app/startup.py"]