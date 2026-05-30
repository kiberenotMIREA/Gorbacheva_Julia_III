# Vulnerability Scoring API

Обнаружение уязвимостей C/C++ кода с использованием ансамбля моделей (BERT + LSTM).

---

## 1. Паспорт проекта

- **Название проекта:** Обнаружение уязвимостей C/C++ кода с использованием ансамбля моделей (BERT + LSTM)
- **Автор:** Горбачева Юлия Павловна
- **Группа:** КТСО-12-24
- **Контакт:** @yuliyaenot

- **Краткое описание:** Проект посвящен построению сервиса для автоматического обнаружения уязвимостей в коде C/C++ с использованием ансамбля моделей (BERT + LSTM). Используется датасет VulDeePecker (CWE-119 + CWE-399). Реализованы и сравнены модели (LogisticRegression, BiLSTM, DistilBERT с mean pooling) и ансамбль Stacking (BERT + LSTM, 801-d мета-признаки, F1=0.805). Результат — REST API на FastAPI, возвращающий вероятность наличия уязвимости.

---

## 2. Структура проекта

```
project/
├── requirements.txt          # Зависимости Python (CPU-only, 22 пакета)
├── README.md                 # Паспорт проекта и инструкции по запуску
├── report.md                 # Полный отчет (9 разделов, таблицы, обоснование)
├── Dockerfile                # Многослойная сборка Docker-образа API (CPU-only torch)
├── docker-compose.yml        # Docker Compose (API + Frontend)
├── Makefile                  # Один клик: make setup, make train, make service, make docker, make all
├── .gitignore                # Игнорирование .env, моделей, кэша
├── scripts/
│   ├── setup.sh              # Установка: .env → venv → pip (без запуска API)
│   └── setup.bat             # То же для Windows
│   ├── cleanup.sh            # Удаление артефактов (Linux)
│   └── cleanup.bat           # Удаление артефактов (Windows)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Загрузка HuggingFace, очистка, сплиты, TF-IDF, LSTM токенизация
│   ├── features.py           # Извлечение BERT-эмбеддингов, LSTM-признаков, гибридных признаков
│   ├── models.py             # BiLSTM, LogisticRegression, мета-классификатор
│   ├── train.py              # Полный пайплайн обучения (4 модели, сравнение, сохранение + MLflow)
│   ├── predict.py            # Класс VulnerabilityPredictor (bert/ensemble методы)
│   ├── metrics.py            # Prometheus метрики (REQUEST_COUNT, REQUEST_LATENCY, MODEL_STATUS, ERROR_COUNT, INPUT_SIZE)
│   └── service.py            # FastAPI сервис (/predict, /health, /methods, /metrics, логирование, Prometheus)
├── notebooks/
│   └── 00_full_pipeline.ipynb # Полный пайплайн: EDA, TF-IDF, BiLSTM, DistilBERT, Ensemble, сравнение
├── frontend/
│   ├── Dockerfile            # Многослойная сборка React (Vite + Nginx)
│   ├── nginx/default.conf    # Nginx конфиг с прокси на API
│   ├── src/App.jsx           # React компонент (UI: предсказание, методы, статус)
│   └── ...                   # Vite, package.json, index.html
├── configs/
│   ├── config.yaml           # Конфигурация (данные, модели, сервис)
│   └── .env.example          # Шаблон переменных окружения (без секретов)
├── tests/
│   ├── __init__.py
│   ├── test_data.py          # 5 тестов на загрузку и обработку данных
│   ├── test_features.py      # 3 теста на BERT и гибридные признаки
│   ├── test_predict.py       # 4 теста на инференс (bert/ensemble, без моделей)
│   └── test_service.py       # 5 тестов на API (health, predict, валидация)
├── data/
│   └── processed/            # Обработанные данные (pickle)
└── artifacts/
    ├── models/               # Сохраненные модели (.keras, .pkl, .json)
    ├── figures/              # Графики и визуализации
    └── logs/                 # Логи обучения (TensorBoard)
```

---

## 3. Требования и установка

### 3.1. Системные требования

- **Python** >= 3.10
- **ОС:** Windows 10 / Debian 13 (тестировано на обеих платформах)
- **Оперативная память:** >= 8 GB (рекомендуется 16 GB для DistilBERT)
- **Интернет:** требуется при первом запуске (загрузка датасета и модели DistilBERT)

### 3.2. Установка окружения

```bash
# Перейти в папку проекта
cd project

# Создать виртуальное окружение
python -m venv .venv

# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Обновить pip
pip install --upgrade pip

# Установить CPU-only PyTorch (120MB вместо 2.4GB с CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

# Установить остальные зависимости (transformers увидит уже установленный torch)
pip install -r requirements.txt

# Скопировать шаблон переменных окружения
cp configs/.env.example .env
```

> **Примечание:** Если на Linux TensorFlow < 2.16, используйте `pip install tensorflow-cpu` вместо `tensorflow`.

### 3.3. Быстрый старт (скрипты в один клик)

> **Примечание:** Перед запуском скриптов сделайте их исполняемыми:
> ```bash
> chmod +x scripts/*.sh
> ```

```bash
# Linux — установка зависимостей (без запуска API):
./scripts/setup.sh

# Linux — установка и обучение моделей:
./scripts/setup.sh train

# Linux — сборка и запуск Docker (блокирующий):
./scripts/setup.sh docker

# Linux — сборка и запуск Docker (фоновый):
./scripts/setup.sh docker --detach

# Windows:
scripts\setup.bat
scripts\setup.bat train
scripts\setup.bat docker
scripts\setup.bat docker --detach
```

**Или через Makefile (Linux):**

```bash
make setup       # установка окружения
make train       # обучение моделей
make service     # запуск API сервиса
make docker      # сборка и запуск Docker (фоново)
make mlflow      # запуск MLflow UI (http://localhost:5000)
make test        # запуск тестов
make all         # полный цикл: setup → train → docker
```

Скрипты автоматически:
1. Создают `.env` из шаблона (если нет)
2. Создают виртуальное окружение `.venv` (только для `train`/`service`)
3. Устанавливают CPU-only PyTorch (первым, чтобы transformers не тянул CUDA-версию)
4. Устанавливают остальные зависимости
5. Запускают сервис / обучение / Docker-контейнеры

---

> **Замечание по адресам:** Во всех примерах ниже используется `localhost`. Если сервисы запущены на удаленном сервере, заменяйте `localhost` на его IP-адрес (например, `192.168.137.178`). При запуске через Docker контейнеры публикуют порты на `0.0.0.0`, так что сервис будет доступен извне.

## 4. Как запустить проект

### 4.1. Обучение моделей

```bash
cd project
python -m src.train
```

Пайплайн обучения выполнит:
1. Загрузку датасета VulDeePecker (~3500 примеров)
2. Очистку (дедупликация, фильтрация по длине)
3. Разделение на train/val/test (70/15/15)
4. Обучение 4 моделей: LR, BiLSTM, DistilBERT+LR (mean pooling), ансамбль Stacking (BERT + LSTM, 801-d meta)
5. Сравнение метрик и определение лучшей модели
6. Сохранение всех моделей в `artifacts/models/`

> **Время выполнения:** ~30-75 минут (зависит от CPU; fine-tuning DistilBERT добавляет ~15 мин)

### 4.2. Запуск сервиса (локально)

```bash
cd project
python -m src.service
```

Сервис запускается на `http://localhost:8000` (при запуске на удаленном сервере используйте его IP-адрес вместо localhost)

**Frontend (локальная разработка):**

```bash
cd project/frontend
npm install
npm run dev        # Режим разработки на http://localhost:5173
```

### 4.3. Запуск через Docker

**Рекомендуемый способ (скрипт):**

```bash
# Linux (блокирующий режим):
./scripts/setup.sh docker

# Linux (фоновый режим):
./scripts/setup.sh docker --detach

# Windows:
scripts\setup.bat docker
scripts\setup.bat docker --detach
```

**Или через Makefile:**
```bash
make docker      # фоновый запуск Docker (docker compose up --build -d)
make docker-logs # блокирующий запуск с логами
make docker-stop # остановка контейнеров
```

**Или напрямую через Docker Compose:**

```bash
cd project
docker compose up --build
```

Эта команда собирает и запускает два контейнера:
- **vulnerability-api** — FastAPI бэкенд (порт 8000)
- **frontend** — React frontend через Nginx (порт 3000, прокси `/api/*` на бэкенд)

Откройте в браузере: `http://localhost:3000` (на удаленном сервере — `http://<IP-адрес сервера>:3000`)

Для запуска только API без frontend:

```bash
docker compose up --build vulnerability-api
```

### 4.4. Эндпоинты API

| Эндпоинт | Метод | Описание | Статус-коды |
|-----------|-------|----------|-------------|
| `/health` | GET | Проверка состояния сервиса и загрузки моделей | 200 |
| `/methods` | GET | Список доступных методов обнаружения дефектов | 200 |
| `/predict` | POST | Предсказание уязвимости по фрагменту кода | 200, 422, 500, 503 |
| `/metrics` | GET | Prometheus-метрики для мониторинга | 200 |

### 4.5. Примеры запросов

Во всех примерах используется `localhost`. При обращении к удаленному серверу замените `localhost` на его IP-адрес.

**Проверка здоровья сервиса:**

```bash
curl http://localhost:8000/health
```

Ответ:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "artifacts/models",
  "version": "1.0.0"
}
```

**Список доступных методов:**
```bash
curl http://localhost:8000/methods
```

Ответ:
```json
{
  "methods": [
    {
      "id": "ensemble",
      "name": "Ансамбль Stacking (BERT + LSTM)",
      "description": "Мета-модель: BERT_LR(1-d) + BERT_mean(768-d) + LSTM(32-d) → мета-классификатор (801-d).",
      "base_models": ["DistilBERT", "LogisticRegression", "Bidirectional LSTM"]
    },
    {
      "id": "bert",
      "name": "DistilBERT + LogisticRegression (mean pooling)",
      "description": "Mean pooling эмбеддинги (768-d) + L2-нормализация → LogisticRegression",
      "base_models": ["DistilBERT", "LogisticRegression"]
    }
  ],
  "default": "ensemble"
}
```

**Предсказание уязвимости (ансамбль Stacking — метод по умолчанию):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "void func() { char buf[10]; strcpy(buf, input); }",
    "method": "ensemble"
  }'
```

Ответ:
```json
{
  "prediction": 1,
  "label": "vulnerable",
  "probabilities": [0.214, 0.786],
  "confidence": 0.786,
  "inference_time_ms": 2450.32,
  "method": "ensemble"
}
```

**Безопасный код:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "int add(int a, int b) { return a + b; }",
    "method": "ensemble"
  }'
```

Ответ:
```json
{
  "prediction": 0,
  "label": "safe",
  "probabilities": [0.952, 0.048],
  "confidence": 0.952,
  "inference_time_ms": 2310.45,
  "method": "ensemble"
}
```

### 4.6. Интерактивная документация (Swagger UI)

После запуска сервиса откройте в браузере: `http://localhost:8000/docs` (на удаленном сервере — `http://<IP-адрес сервера>:8000/docs`)

---

## 5. Данные

### VulDeePecker (источник данных)

- **Название:** VulDeePecker [Li et al., 2018]
- **Платформа:** HuggingFace Datasets (`claudios/VulDeePecker`)
- **Размер:** ~160,000 функций C/C++ из opensource-проектов
- **Разметка:** CWE-119 (ошибки буфера) + CWE-399 (ошибки управления ресурсами)
- **Классы:** 0 — безопасный, 1 — уязвимый
- **Дисбаланс:** ~5.7% уязвимых, ~94.3% безопасных

### Предобработка

| Этап | Описание | Удалено |
|------|----------|---------|
| Дедупликация | Удаление одинаковых функций | 398 (11.4%) |
| Фильтрация | 20-2000 символов | 243 (7.8%) |
| **Итог** | Чистый датасет | **2859 примеров** |

### Разделение

| Выборка | Доля | Размер |
|---------|------|--------|
| Train | 70% | 2002 |
| Validation | 15% | 428 |
| Test | 15% | 429 |

---

## 6. Модели

### Сравнение (тестовые метрики)

| Модель | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| LogisticRegression | 0.932 | 0.494 | 0.864 | 0.628 | 0.969 |
| BiLSTM | 0.904 | 0.391 | 0.773 | 0.519 | 0.891 |
| DistilBERT+LR (mean pool) | 0.942 | 0.565 | 0.591 | 0.578 | 0.936 |
| **Ensemble Stacking** (801-d) | **0.976** | **0.868** | **0.750** | **0.805** | 0.946 |

### Финальная модель: Ensemble Stacking (BERT + LSTM, 801-d meta, F1=0.805)

**Обоснование выбора:**
- **F1=0.805** — наивысший среди всех моделей
- **Precision=0.868** — надежные предсказания с минимальным FP
- **Recall=0.750** — высокий охват уязвимостей
- **801-d мета-признаки:** BERT mean embedding (768-d) напрямую + LSTM (32-d) + BERT-LR proba (1-d)
- **Mean pooling BERT** + L2-нормализация — более робастное представление кода

---

## 7. Тесты, трекинг и метрики

### 7.1. Тесты

```bash
cd project
pytest tests -v
```

Тесты покрывают (всего **18 тестов**):

**Модульные тесты данных** (`tests/test_data.py` — 5 тестов):
1. Дедупликация — `test_clean_data_removes_duplicates`
2. Фильтрация по длине — `test_clean_data_filters_by_length`
3. Разделение на сплиты — `test_prepare_splits_maintains_proportions`
4. TF-IDF признаки — `test_create_tfidf_features`
5. LSTM последовательности — `test_create_lstm_sequences`

**Тесты признаков** (`tests/test_features.py` — 3 теста):
1. Гибридные признаки — `test_build_hybrid_features_shapes`
2. Несовпадение размерностей — `test_build_hybrid_features_mismatch`
3. Загрузка BERT — `test_load_bert_models`

**Тесты инференса** (`tests/test_predict.py` — 4 теста):
1. Без моделей — `test_predictor_no_models`
2. Неизвестный метод — `test_predict_unknown_method`
3. BERT без моделей — `test_predict_bert_no_models`
4. Ensemble без моделей — `test_predict_ensemble_no_models`

**Тесты сервиса** (`tests/test_service.py` — 6 тестов):
1. Health-check — `test_health_endpoint`
2. Пустое тело — `test_predict_endpoint_empty_body`
3. Короткий код — `test_predict_endpoint_invalid_code`
4. Валидный запрос — `test_predict_endpoint_valid_request`
5. BERT метод — `test_predict_endpoint_bert_method`
6. Prometheus метрики — `test_metrics_endpoint`

### 7.2. MLflow Experiment Tracking

Во время обучения (`python -m src.train`) автоматически логируются:

**Параметры:**
- `sample_size`, `test_size`, `val_size` — параметры данных
- `lstm_epochs`, `lstm_batch_size`, `lstm_units` — гиперпараметры LSTM
- `bert_model_name`, `max_length` — параметры BERT
- `tfidf_max_features`, `tfidf_ngram_range` — параметры TF-IDF

**Метрики (на тестовой выборке):**
- Для каждой из 4 моделей: `{model}_f1`, `{model}_precision`, `{model}_recall`, `{model}_accuracy`, `{model}_roc_auc`

**Артефакты:**
- Все сохраненные модели (`artifacts/models/`)
- CSV-файл сравнения моделей (`artifacts/comparison_results.csv`)
- Конфигурация (`configs/config.yaml`)

**Просмотр:**
```bash
make mlflow
# или
mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db
# Открыть http://localhost:5000 (на удаленном сервере — http://<IP-адрес сервера>:5000)
```

### 7.3. Prometheus Метрики

На эндпоинте `GET /metrics` доступны метрики в формате Prometheus:

| Метрика | Тип | Описание |
|---------|-----|----------|
| `vulnerability_predict_requests_total` | Counter | Количество запросов по методу и метке |
| `vulnerability_predict_latency_seconds` | Histogram | Гистограмма времени инференса |
| `vulnerability_model_loaded` | Gauge | Статус загрузки модели (1 — загружена, 0 — нет) |
| `vulnerability_errors_total` | Counter | Количество ошибок по типу |
| `vulnerability_input_size_bytes` | Histogram | Размер входного кода в байтах |

Пример запроса:
```bash
curl http://localhost:8000/metrics
```

---

## 8. Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    VulDeePecker Dataset                      │
│              (HuggingFace: claudios/VulDeePecker)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Preprocessing Pipeline                          │
│  1. Дедупликация   2. Фильтрация 20-2000   3. Train/Val/Test│
└────┬────────┬──────────┬────────────────┬───────────────────┘
     │        │          │                │
      ▼        ▼          ▼
┌────────┐ ┌────────┐    ┌────────────┐
│ TF-IDF │ │ Char   │    │ DistilBERT │
│ → LR   │ │ → LSTM │    │ → mean(768)│
└────────┘ └───┬────┘    └──────┬─────┘
               │                │
               └────────┬───────┘
                        │
                        ▼
     ┌────────────────────────────────────────┐
      │     Ensemble Stacking                     │
      │  BERT_LR(1-d) + BERT_mean(768-d) +     │
      │  LSTM(32-d) → Meta (801-d)              │
     └──────────────────┬─────────────────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │   FastAPI Service    │
          │  /predict  /health   │
          │  /methods  /metrics  │
          └──────────┬───────────┘
                     │
                     ▼
          ┌─────────────────────┐
           │  MLflow Tracking                   │
           │  artifacts/mlruns/mlflow.db      │
           └──────────────────────────────────┘
```

---

## 9. Демонстрация на защите

1. **Структура проекта** — покажу организацию кода: `notebooks/`, `src/`, `configs/`, `tests/`, `frontend/`, `Dockerfile`
2. **Запуск сервиса** — `./scripts/setup.sh service` и открою Swagger UI (`http://localhost:8000/docs`)
3. **Ключевые сценарии:**
   - Безопасный код → `/predict` возвращает `"label": "safe"`
   - Уязвимый код (strcpy без проверки) → `/predict` возвращает `"label": "vulnerable"`
   - `/health` → проверка состояния сервиса
4. **Результаты экспериментов:**
   - EDA: распределение классов (5.7% уязвимых), очистка данных
    - Сравнение 4 моделей: таблица метрик, ROC-кривые, матрицы ошибок
     - Обоснование выбора ансамблевой модели Stacking (F1=0.805)
5. **Тесты:** запущу `pytest tests -v`
6. **Docker:** покажу сборку и запуск через `./scripts/setup.sh docker`

---

## 10. Ограничения и дальнейшая работа

### Текущие ограничения
- Модель обучена на подвыборке (3000 из 160,000) из-за CPU-only
- DistilBERT в режиме feature extraction (fine-tuning опционален, ~15 мин)
- LSTM обучен с EarlyStopping (обычно 1-3 эпохи)
- Нет авторизации/аутентификации
- Нет персистентного хранения истории запросов
- Нет Grafana для визуализации Prometheus метрик

### Перспективы развития
1. **Модель:** увеличение эпох LSTM, Optuna для подбора гиперпараметров
2. **Данные:** обучение на полном датасете, аугментация кода
3. **Архитектура:** многоклассовая классификация по CWE
4. **Оптимизация:** квантование DistilBERT, ONNX Runtime
5. **MLOps:** интеграция Grafana для визуализации Prometheus метрик, CI/CD, авторизация

---

## 11. Переменные окружения (CPU-only)

Для стабильной работы на CPU установите в `.env`:

| Переменная | Значение | Назначение |
|-----------|----------|------------|
| `CUDA_VISIBLE_DEVICES` | `-1` | Отключает поиск GPU (подавляет CUDA-warnings) |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Отключает oneDNN-оптимизации (стабильность) |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Подавляет INFO-сообщения TensorFlow |
| `TRACKING_URI` | `sqlite:///artifacts/mlruns/mlflow.db` | Путь к хранилищу MLflow экспериментов |

Эти переменные уже установлены в `Dockerfile`, `docker-compose.yml` и `Makefile`.
