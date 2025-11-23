# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    BOT_CONFIG=/app/bot.config.json

# Install essential development tools for DigitalOcean console debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8080

CMD ["python", "startup.py"]
