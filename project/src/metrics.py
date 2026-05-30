# =============================================================================
# Модуль Prometheus-метрик для наблюдаемости сервиса
# =============================================================================
# Предоставляет счетчики, гистограммы и gauges для мониторинга:
#   - Количество запросов /predict (всего и по методам/меткам)
#   - Гистограмма времени инференса
#   - Статус загрузки модели (0/1)
#   - Эндпоинт /metrics для сбора метрик Prometheus
# =============================================================================

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response


# Счетчик всех запросов к /predict (label method + label)
REQUEST_COUNT = Counter(
    "vulnerability_predict_requests_total",
    "Общее количество запросов к /predict",
    ["method", "label"],
)

# Гистограмма времени инференса в секундах
REQUEST_LATENCY = Histogram(
    "vulnerability_predict_latency_seconds",
    "Гистограмма времени инференса /predict (сек)",
    ["method"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

# Gauge статуса загрузки модели (0 — не загружена, 1 — загружена)
MODEL_STATUS = Gauge(
    "vulnerability_model_loaded",
    "Статус загрузки модели (0 = не загружена, 1 = загружена)",
)

# Счетчик ошибок предсказания
ERROR_COUNT = Counter(
    "vulnerability_predict_errors_total",
    "Общее количество ошибок при предсказании",
    ["method"],
)

# Гистограмма размера входного кода (символы)
INPUT_SIZE = Histogram(
    "vulnerability_input_code_length_bytes",
    "Гистограмма длины входного кода (символы)",
    buckets=(10, 50, 100, 200, 500, 1000, 2000, 5000, float("inf")),
)


def get_metrics_response() -> Response:
    """
    Формирование HTTP-ответа с метриками в формате Prometheus.
    Content-Type: text/plain; version=0.0.4
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
