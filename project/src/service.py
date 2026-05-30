# =============================================================================
# FastAPI сервис для Vulnerability Scoring API
# =============================================================================
# Этот модуль реализует REST API для обнаружения уязвимостей в коде C/C++.
#
# Эндпоинты:
#   GET  /health   — проверка состояния сервиса (наблюдаемость)
#   GET  /methods  — список доступных методов обнаружения дефектов
#   POST /predict  — предсказание уязвимости по фрагменту кода
#   GET  /metrics  — Prometheus метрики для мониторинга
#
# Функциональность:
#   - Загрузка моделей при старте сервиса (через lifespan)
#   - Логирование всех запросов и ошибок (через logging)
#   - Измерение времени инференса
#   - Конфигурация через переменные окружения (.env)
#   - Валидация входных данных через Pydantic
# =============================================================================

import os  # Доступ к переменным окружения
import logging  # Логирование событий и ошибок
import time  # Измерение времени инференса
from contextlib import asynccontextmanager  # Управление жизненным циклом приложения

from fastapi import FastAPI, HTTPException  # Веб-фреймворк FastAPI
from pydantic import BaseModel, Field  # Валидация данных (Pydantic)
from dotenv import load_dotenv  # Загрузка переменных из .env файла

import sys
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.predict import VulnerabilityPredictor  # Наш класс для предсказаний
from src.metrics import (  # Prometheus-метрики для наблюдаемости
    REQUEST_COUNT,
    REQUEST_LATENCY,
    MODEL_STATUS,
    ERROR_COUNT,
    INPUT_SIZE,
    get_metrics_response,
)

# Загружаем переменные окружения из .env файла (если он существует)
load_dotenv()

# Настройки из переменных окружения с значениями по умолчанию
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Уровень логирования
MODEL_DIR = os.getenv("MODEL_DIR", "artifacts/models")  # Путь к моделям

# Настройка системы логирования: формат с временем, именем модуля, уровнем
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Создаем логгер для нашего сервиса
logger = logging.getLogger("vulnerability-api")

# Глобальная переменная для предсказателя (загружается при старте сервиса)
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом FastAPI приложения.
    
    Выполняется при старте и остановке сервиса.
    Загружаем модели в память при старте. Если загрузка не удалась,
    сервис запускается в 'degraded' режиме — /health вернет статус degraded,
    а /predict будет возвращать ошибку 503.
    
    Используется современный lifespan вместо deprecated @app.on_event.
    """
    global predictor
    logger.info("Загрузка моделей...")
    try:
        predictor = VulnerabilityPredictor(models_dir=MODEL_DIR)
        MODEL_STATUS.set(1)
        logger.info("Модели загружены")
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}")
        MODEL_STATUS.set(0)
        predictor = None
    yield


# Создаем FastAPI приложение с lifespan-обработчиком
app = FastAPI(
    title=os.getenv("API_TITLE", "Vulnerability Scoring API"),
    version=os.getenv("API_VERSION", "1.0.0"),
    description="API for C/C++ code vulnerability detection using hybrid deep learning models",
    lifespan=lifespan,  # Используем lifespan вместо on_event
)


# =============================================================================
# Pydantic модели для валидации входных и выходных данных
# =============================================================================

class CodeInput(BaseModel):
    """
    Модель входных данных для эндпоинта /predict.
    
    Атрибуты:
        code (str): фрагмент кода C/C++ для анализа (мин. 10 символов)
        method (str): метод предсказания:
            - 'ensemble' (по умолчанию): ансамбль Stacking (BERT + LSTM)
            - 'bert': DistilBERT + LogisticRegression
    """
    code: str = Field(
        ...,  # Обязательное поле (три точки = обязательное)
        description="C/C++ source code to analyze",
        min_length=10,  # Минимальная длина кода — 10 символов
    )
    method: str = Field(
        default="ensemble",
        description="Model method: 'ensemble' (Stacking, default) or 'bert' (DistilBERT+LR)",
    )


class PredictionOutput(BaseModel):
    """
    Модель выходных данных эндпоинта /predict.
    
    Атрибуты:
        prediction (int): 0 — безопасный код, 1 — уязвимый
        label (str): текстовое описание ('safe' или 'vulnerable')
        probabilities (list): вероятности [p_safe, p_vulnerable]
        confidence (float): уверенность модели (максимальная вероятность, 0-1)
        inference_time_ms (float): время инференса в миллисекундах
        method (str): использованный метод предсказания
    """
    prediction: int  # 0 (safe) или 1 (vulnerable)
    label: str  # Текстовое представление для удобства чтения
    probabilities: list  # Вероятности обоих классов [p_safe, p_vuln]
    confidence: float  # Максимальная вероятность (уверенность модели)
    inference_time_ms: float  # Время обработки запроса в мс
    method: str  # Использованный метод ('hybrid', 'bert', 'ensemble')


class MethodInfo(BaseModel):
    """
    Модель описания одного метода обнаружения дефектов.
    
    Атрибуты:
        id (str): уникальный идентификатор метода (для передачи в /predict)
        name (str): человекочитаемое название
        description (str): подробное описание архитектуры и принципа работы
        base_models (list): список базовых моделей, используемых методом
    """
    id: str
    name: str
    description: str
    base_models: list


class MethodsOutput(BaseModel):
    """
    Модель выходных данных эндпоинта /methods.
    
    Атрибуты:
        methods (list[MethodInfo]): список доступных методов с описанием
        default (str): метод по умолчанию (используется, если не указан в /predict)
    """
    methods: list
    default: str


class HealthOutput(BaseModel):
    """
    Модель выходных данных эндпоинта /health.
    
    Атрибуты:
        status (str): 'healthy' — модели загружены, 'degraded' — нет
        model_loaded (bool): загружены ли модели в память
        model_path (str): путь к директории с моделями
        version (str): версия API
    """
    status: str  # Статус сервиса
    model_loaded: bool  # Флаг загрузки модели
    model_path: str  # Путь к моделям
    version: str  # Версия API


# =============================================================================
# Эндпоинты API
# =============================================================================

@app.get("/health", response_model=HealthOutput)
async def health():
    """
    GET /health — эндпоинт проверки состояния сервиса.
    
    Используется для:
    - Мониторинга (проверка, что сервис жив)
    - Kubernetes/Docker healthcheck probes
    - Диагностики (загружены ли модели)
    
    Возвращает статус сервиса и информацию о загрузке моделей.
    Всегда возвращает 200 (сервис запущен), но статус может быть
    'degraded', если модели не загружены.
    """
    return HealthOutput(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_path=MODEL_DIR,
        version="1.0.0",
    )


@app.get("/metrics")
async def metrics():
    """
    GET /metrics — эндпоинт для сбора Prometheus-метрик.
    
    Возвращает метрики в формате Prometheus text/plain.
    Используется Prometheus server для скрейпинга.
    """
    return get_metrics_response()


# Статическая информация о доступных методах обнаружения дефектов.
# Каждый метод имеет уникальный id, название, описание архитектуры
# и список базовых моделей, которые он использует.
AVAILABLE_METHODS = [
    MethodInfo(
        id="ensemble",
        name="Ансамбль Stacking (BERT + LSTM)",
        description="Мета-модель на 801-d мета-признаках: вероятность "
                    "DistilBERT+LR (1-d), mean-pooling BERT-эмбеддинги "
                    "(768-d) и LSTM penultimate признаки (32-d). "
                    "F1-score: 0.805.",
        base_models=["DistilBERT",
                     "LogisticRegression", "Bidirectional LSTM"],
    ),
    MethodInfo(
        id="bert",
        name="DistilBERT + LogisticRegression (mean pooling)",
        description="Извлечение mean-pooling эмбеддингов (768-d) из "
                    "DistilBERT с L2-нормализацией. Эмбеддинги подаются "
                    "на LogisticRegression с class_weight='balanced'. "
                    "Mean pooling дает более робастное представление "
                    "для кода, чем [CLS]-токен.",
        base_models=["DistilBERT", "LogisticRegression"],
    ),
]


@app.get("/methods", response_model=MethodsOutput)
async def list_methods():
    """
    GET /methods — список доступных методов обнаружения дефектов.
    
    Возвращает информацию о каждом методе: id (для передачи в /predict),
    название, подробное описание архитектуры и список базовых моделей.
    
    Позволяет клиентам динамически узнавать, какие методы доступны,
    без обращения к документации.
    """
    return MethodsOutput(methods=AVAILABLE_METHODS, default="ensemble")


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: CodeInput):
    """
    POST /predict — предсказание уязвимости в коде.
    
    Принимает фрагмент кода C/C++ и возвращает вероятность наличия уязвимости.
    
    Параметры запроса (JSON body):
        code (str): исходный код для анализа (мин. 10 символов)
        method (str): метод предсказания:
            - 'ensemble' (по умолчанию): ансамбль Stacking (BERT + LSTM)
            - 'bert': DistilBERT + LogisticRegression
    
    Возвращает:
        PredictionOutput с предсказанием, вероятностями и временем инференса
    
    Коды ошибок:
        - 422: невалидные входные данные (короткий код, неизвестный метод)
        - 503: модели не загружены (сервис в degraded режиме)
        - 500: внутренняя ошибка при предсказании
    """
    # Проверяем, загружены ли модели
    if predictor is None:
        ERROR_COUNT.labels(method=input_data.method).inc()
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Сначала обучите модели: python -m src.train"
        )

    # Валидация метода предсказания: разрешены только 'bert', 'ensemble'
    valid_methods = {"bert", "ensemble"}
    if input_data.method not in valid_methods:
        raise HTTPException(
            status_code=422,
            detail=f"Неверный метод '{input_data.method}'. Допустимые методы: {', '.join(sorted(valid_methods))}"
        )

    # Логируем входящий запрос: размер кода и выбранный метод
    logger.info(f"Получен запрос predict (метод={input_data.method}, "
                f"длина_кода={len(input_data.code)})")
    INPUT_SIZE.observe(len(input_data.code))

    # Засекаем время начала инференса
    start = time.time()

    # Выполняем предсказание через выбранный метод
    try:
        pred, proba = predictor.predict(input_data.code, method=input_data.method)
    except Exception as e:
        ERROR_COUNT.labels(method=input_data.method).inc()
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

    # Вычисляем время инференса в миллисекундах
    elapsed = (time.time() - start) * 1000

    # Если предсказание не удалось (компоненты модели не загружены)
    if pred is None:
        ERROR_COUNT.labels(method=input_data.method).inc()
        raise HTTPException(
            status_code=503,
            detail="Компоненты модели загружены не полностью. Убедитесь, что все модели обучены.",
        )

    # Преобразуем числовое предсказание в текст
    label = "vulnerable" if pred == 1 else "safe"
    confidence = max(proba) if proba else 0.0

    # Записываем Prometheus-метрики
    REQUEST_COUNT.labels(method=input_data.method, label=label).inc()
    REQUEST_LATENCY.labels(method=input_data.method).observe(elapsed / 1000.0)

    # Логируем результат предсказания
    logger.info(f"Предсказание: {label} (уверенность={confidence:.3f}, "
                f"время={elapsed:.1f}мс)")

    # Возвращаем структурированный ответ
    return PredictionOutput(
        prediction=pred,
        label=label,
        probabilities=proba,
        confidence=confidence,
        inference_time_ms=round(elapsed, 2),  # Округляем до 2 знаков
        method=input_data.method,
    )


# Точка входа: запуск Uvicorn сервера
if __name__ == "__main__":
    import uvicorn  # ASGI-сервер для FastAPI

    # Читаем настройки хоста и порта из переменных окружения
    host = os.getenv("SERVICE_HOST", "0.0.0.0")  # По умолчанию принимаем все подключения
    port = int(os.getenv("SERVICE_PORT", "8000"))  # По умолчанию порт 8000

    # Запускаем сервер Uvicorn
    uvicorn.run(app, host=host, port=port)
