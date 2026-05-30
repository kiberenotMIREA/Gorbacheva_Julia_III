# Самопроверка проекта (self-checklist)

Чеклист для самопроверки проекта перед сдачей. Отмечены пункты, реализованные в проекте.

---

## Таблица самопроверки

| #  | Критерий | Да/Нет | Где смотреть / комментарий |
|----|----------|--------|---------------------------|
| 1 | Сервис запускается по инструкциям из `project/README.md` и работает | ✅ | `README.md`, раздел «Как запустить проект». FastAPI сервис запускается через `python -m src.service` (локально) или `docker compose up --build` (Docker, API + Frontend). Работает на `http://localhost:8000` (API) / `http://localhost:3000` (UI). |
| 2 | Endpoint `/predict` использует **реальную модель**, а не заглушку | ✅ | `src/predict.py` — класс `VulnerabilityPredictor` загружает реальные обученные модели: LSTM (`lstm_model.keras`), DistilBERT (HuggingFace), Ensemble (`ensemble_meta_clf.pkl`). `src/service.py` — FastAPI использует `predictor.predict()` для методов bert/ensemble. |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | ✅ | `notebooks/00_full_pipeline.ipynb` — EDA (распределение классов, длины кода, дедупликация, фильтрация), эксперименты с LR, BiLSTM, DistilBERT, Ensemble Stacking, ROC-кривые, матрицы ошибок. |
| 4 | Есть baseline и улучшенная модель, есть **сравнение по метрикам** | ✅ | Baseline: LogisticRegression (F1=0.628). Улучшенная: Ensemble Stacking (F1=0.805, Precision=0.868, Recall=0.750). Сравнение 4 моделей (LR, BiLSTM, DistilBERT+LR, Ensemble) в `report.md` (таблица 3) и `notebooks/00_full_pipeline.ipynb` (ROC-кривые, матрицы ошибок). |
| 5 | Код не свален в один ноутбук: есть внятная структура в `src/` | ✅ | Четкое разделение: `data_loader.py` (данные), `features.py` (признаки), `models.py` (модели), `train.py` (пайплайн), `predict.py` (инференс), `service.py` (API), `scripts/setup.sh` (один клик). Каждый модуль с комментариями на русском. |
| 6 | Есть Dockerfile **или** понятный сценарий развертывания без Docker | ✅ | `Dockerfile` (многослойная сборка) + `docker-compose.yml` (API + Frontend). `frontend/Dockerfile` (React + Nginx). Скрипты `scripts/setup.sh` / `setup.bat` — один клик для Debian и Windows. Подробные инструкции в `README.md` раздел 4. |
| 7 | Есть `.env.example` и **нет** в репозитории реальных секретов/паролей | ✅ | `configs/.env.example` — шаблон с 18 переменными без значений. `.env` добавлен в `.gitignore`. В репозитории нет реальных секретов, токенов или паролей. |
| 8 | Реализованы логи/наблюдаемость (консольные логи + `/health`) + MLflow трекинг экспериментов + Prometheus метрики (`/metrics`) | ✅ | `src/service.py`: консольные логи входящих запросов и результатов. Endpoint `GET /health`. `src/train.py`: MLflow логирование параметров, метрик и артефактов. `src/metrics.py`: 5 Prometheus метрик. `GET /metrics` на порту 8000. `sqlite:///artifacts/mlruns/mlflow.db` — SQLite-хранилище MLflow. `Makefile`: `make mlflow` → `mlflow ui`. |
| 9 | В `report.md` **обоснован выбор финальной модели** по результатам экспериментов | ✅ | `report.md`, раздел 5: таблица сравнения 4 моделей по 5 метрикам, таблица ошибок (FP/FN), обоснование выбора Ensemble Stacking (F1=0.805, Precision=0.868, Recall=0.750), 801-d meta признаки, mean pooling BERT. |
| 10 | `project/README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | `README.md`, раздел 9 «Демонстрация на защите» — 6 шагов демонстрации, включая Docker, Swagger UI, frontend. `report.md`, раздел 9 «Сценарий демонстрации на защите» — детальный сценарий с командами и упоминанием MLflow UI. Скрипт `scripts/setup.sh` — запуск в один клик. |

---

## Подсчет баллов

- **Реализовано критериев:** 10 из 10 ✅
