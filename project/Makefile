# =============================================================================
# Makefile для проекта Vulnerability Scoring
# =============================================================================
# Цели:
#   make setup     — создать .env + venv + установить зависимости (без запуска API)
#   make train     — обучить модели (локально)
#   make service   — запустить API сервис (локально)
#   make docker    — собрать и запустить Docker-контейнеры (фоново)
#   make docker-logs — запустить Docker-контейнеры (блокирующе, с логами)
#   make docker-stop — остановить Docker-контейнеры
#   make mlflow    — запустить MLflow UI для просмотра экспериментов
#   make test      — запустить тесты
#   make all       — полный цикл: установка + обучение + Docker
# =============================================================================

SHELL := /bin/bash

.PHONY: setup train service docker docker-logs docker-stop mlflow test all

setup:
	./scripts/setup.sh

train:
	./scripts/setup.sh train

service:
	./scripts/setup.sh service

docker:
	./scripts/setup.sh docker --detach

docker-logs:
	./scripts/setup.sh docker

docker-stop:
	docker compose down

mlflow:
	@echo "Starting MLflow UI at http://0.0.0.0:5000 ..."
	@.venv/bin/mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db

test:
	pytest tests -v

all: setup train docker
	@echo "=== Done: setup + train + docker ==="
