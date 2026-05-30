# =============================================================================
# Многостадийный Docker-образ для Vulnerability Scoring API
# =============================================================================
# Стадия 1 (builder): установка Python-зависимостей
# Стадия 2 (runtime): минимальный образ с кодом и моделями
#
# Сборка:
#   docker build -t vulnerability-scoring-api .
# Запуск:
#   docker run -p 8000:8000 -v %cd%/artifacts:/app/artifacts vulnerability-scoring-api
# =============================================================================

# --- Стадия 1: Установка зависимостей ----------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Устанавливаем системные пакеты, необходимые для сборки некоторых Python-пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements.txt для кэширования слоя зависимостей
COPY requirements.txt .

# Устанавливаем Python-зависимости (без кэша для уменьшения размера)
# CPU-only PyTorch: ставим ДО -r requirements.txt, чтобы transformers
# не тянул CUDA-версию torch (2GB) через зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps torch && \
    pip install --no-cache-dir -r requirements.txt

# --- Стадия 2: Финальный образ -----------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Устанавливаем curl для Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем установленные пакеты из стадии builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем исходный код и конфигурацию проекта
COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY artifacts/ artifacts/
COPY .env .env

# Открываем порт для FastAPI
EXPOSE 8000

# Переменные окружения для сервиса
ENV PYTHONPATH=/app
ENV SERVICE_HOST=0.0.0.0
ENV SERVICE_PORT=8000
ENV MODEL_DIR=artifacts/models
ENV LOG_LEVEL=INFO

# Отключаем GPU (CPU-only) — подавляет предупреждения CUDA/cuDNN
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

# Точка входа: запуск FastAPI через Uvicorn
CMD ["python", "-m", "src.service"]
