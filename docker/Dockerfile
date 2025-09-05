# Build image
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src ./src
COPY api ./api
COPY configs ./configs
COPY artifacts ./artifacts

EXPOSE 8000
ENV MODEL_PATH=/app/artifacts/model.onnx \
    TOKENIZER_DIR=/app/artifacts/hf

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
