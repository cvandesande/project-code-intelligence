FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_CODE_INTELLIGENCE_FASTEMBED_HOST=0.0.0.0 \
    PROJECT_CODE_INTELLIGENCE_FASTEMBED_CACHE_DIR=/models/fastembed

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir '.[local-embeddings]'

EXPOSE 18081

ENTRYPOINT ["pci-fastembed-server"]
