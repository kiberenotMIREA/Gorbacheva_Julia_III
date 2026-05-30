#!/usr/bin/env bash
# ==============================================================================
# One-click setup and launch for Vulnerability Scoring (Debian / Linux)
# ==============================================================================
# Usage:
#   ./scripts/setup.sh                  — setup venv + install deps (default)
#   ./scripts/setup.sh train            — setup + train models
#   ./scripts/setup.sh service          — setup + start API
#   ./scripts/setup.sh docker           — build and start Docker containers
#   ./scripts/setup.sh docker --detach  — build and start Docker containers in background
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"
ACTION="${1:-}"
DOCKER_FLAG="${2:-}"

echo "========================================================================="
echo "  Vulnerability Scoring — Установка и запуск"
echo "========================================================================="
echo ""

# ------------------------------------------------------------------
# Step 1: Create .env if missing
# ------------------------------------------------------------------
if [ ! -f ".env" ]; then
    echo "[1/5] Создание .env из шаблона..."
    cp configs/.env.example .env
    echo "   OK .env создан"
else
    echo "[1/5] .env уже существует, пропускаем"
fi

# ------------------------------------------------------------------
# Docker path: skip venv/pip, build and launch containers directly
# ------------------------------------------------------------------
if [ "$ACTION" = "docker" ]; then
    echo "[2/2] Сборка и запуск Docker-контейнеров..."
    if ! command -v docker &> /dev/null; then
        echo "ОШИБКА: Docker не найден. Установите Docker."
        exit 1
    fi
    if ! docker compose version &> /dev/null; then
        echo "ОШИБКА: docker compose не доступен."
        exit 1
    fi
    if [ "$DOCKER_FLAG" = "--detach" ]; then
        docker compose up --build -d
    else
        docker compose up --build
    fi
    exit 0
fi

# ------------------------------------------------------------------
# Step 2: Create virtual environment
# ------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "[2/5] Создание виртуального окружения..."
    python3 -m venv .venv
    echo "   OK venv создан"
else
    echo "[2/5] Виртуальное окружение уже существует, пропускаем"
fi

# ------------------------------------------------------------------
# Step 3: Install CPU-only PyTorch FIRST (чтобы transformers не тянул CUDA-версию)
# ------------------------------------------------------------------
echo "[3/5] Установка CPU-версии PyTorch..."
.venv/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps torch
echo "   OK torch (CPU) установлен"

# ------------------------------------------------------------------
# Step 4: Install remaining dependencies
# ------------------------------------------------------------------
echo "[4/5] Установка Python-зависимостей..."
.venv/bin/pip install --quiet -r requirements.txt
echo "   OK зависимости установлены"

# ------------------------------------------------------------------
# Step 5: Run requested action
# ------------------------------------------------------------------
if [ "$ACTION" = "train" ]; then
    echo "[5/5] Обучение моделей..."
    CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
        .venv/bin/python -m src.train
elif [ "$ACTION" = "service" ]; then
    echo "[5/5] Запуск API-сервиса..."
    export LANG=ru_RU.UTF-8 LC_ALL=ru_RU.UTF-8 PYTHONIOENCODING=utf-8
    CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
        .venv/bin/python -m src.service
elif [ -z "$ACTION" ]; then
    echo ""
    echo "========================================================================="
    echo "  Установка завершена. Дальнейшие шаги:"
    echo "    make train     — обучение моделей"
    echo "    make service   — запуск API"
    echo "    make docker    — сборка и запуск Docker"
    echo "========================================================================="
else
    echo "Использование: $0 [train|service|docker]"
    exit 1
fi
