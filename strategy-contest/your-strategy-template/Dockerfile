FROM python:3.11-slim

WORKDIR /app

# Copy base infrastructure and strategy template
COPY base-bot-template/ /app/base-bot-template/
COPY your-strategy-template/ /app/your-strategy-template/

# Install dependencies
RUN pip install --no-cache-dir -r /app/base-bot-template/requirements.txt \
    && pip install --no-cache-dir -r /app/your-strategy-template/requirements.txt

WORKDIR /app/your-strategy-template

EXPOSE 8080 3010

CMD ["python", "startup.py"]
