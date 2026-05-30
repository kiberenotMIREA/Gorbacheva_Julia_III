# =============================================================================
# Тесты для FastAPI сервиса Vulnerability Scoring API
# =============================================================================
# Тесты проверяют:
# 1. GET /health — возвращает правильную структуру ответа
# 2. POST /predict с пустым телом — возвращает 422 (ошибка валидации)
# 3. POST /predict с коротким кодом — возвращает 422
# 4. POST /predict с корректным запросом — возвращает 200 или 503
# 5. POST /predict с методом 'bert' — возвращает 200 или 503
#
# Примечание: сервис может вернуть 503, если модели не обучены.
# Тесты проверяют, что структура ответа корректна в обоих случаях.
# =============================================================================

import sys  # Для добавления пути к корню проекта
import os  # Работа с путями
import pytest  # Фреймворк для тестирования
from fastapi.testclient import TestClient  # Клиент для тестирования FastAPI

# Добавляем корень проекта в путь импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Импортируем FastAPI приложение
from src.service import app


@pytest.fixture
def client():
    """
    Фикстура: создает тестовый HTTP-клиент для FastAPI приложения.
    
    Использование:
        def test_something(client):
            response = client.get("/health")
    
    Возвращает:
        TestClient: тестовый клиент FastAPI
    """
    return TestClient(app)


def test_health_endpoint(client):
    """
    Тест: GET /health возвращает 200 и правильную структуру.
    
    Проверяем наличие всех обязательных полей:
    - status (строка)
    - model_loaded (булево)
    - model_path (строка)
    - version (строка)
    """
    # Отправляем GET запрос к /health
    response = client.get("/health")
    
    # Проверяем статус-код — всегда 200 (сервис запущен)
    assert response.status_code == 200, \
        "Health endpoint should return 200"
    
    # Проверяем структуру ответа
    data = response.json()
    assert "status" in data, "Response should contain 'status'"
    assert "model_loaded" in data, "Response should contain 'model_loaded'"
    assert "model_path" in data, "Response should contain 'model_path'"
    assert "version" in data, "Response should contain 'version'"


def test_predict_endpoint_empty_body(client):
    """
    Тест: POST /predict с пустым телом возвращает 422.
    
    422 Unprocessable Entity — стандартный код FastAPI
    при ошибке валидации Pydantic.
    """
    # Отправляем POST с пустым JSON
    response = client.post("/predict", json={})
    
    # Ожидаем 422 — поле 'code' обязательно
    assert response.status_code == 422, \
        "Empty body should return 422 (validation error)"


def test_predict_endpoint_invalid_code(client):
    """
    Тест: POST /predict со слишком коротким кодом возвращает 422.
    
    Минимальная длина кода — 10 символов (по спецификации).
    """
    # Отправляем код длиной 5 символов
    response = client.post("/predict", json={"code": "short"})
    
    # Ожидаем 422 — не пройдена валидация min_length
    assert response.status_code == 422, \
        "Code shorter than 10 chars should return 422"


def test_predict_endpoint_valid_request(client):
    """
    Тест: POST /predict с корректным запросом.
    
    Может вернуть:
    - 200: если модели загружены (предсказание выполнено)
    - 503: если модели не загружены (см. degraded режим)
    
    В обоих случаях проверяем структуру ответа.
    """
    # Фрагмент кода с уязвимостью (strcpy без проверки длины)
    code = "int main() { char buf[10]; strcpy(buf, input); return 0; }"
    response = client.post("/predict", json={"code": code, "method": "ensemble"})
    
    # Статус может быть 200 (успех) или 503 (модели не загружены)
    assert response.status_code in (200, 503), \
        f"Unexpected status code: {response.status_code}"
    
    # Если предсказание выполнено — проверяем структуру ответа
    if response.status_code == 200:
        data = response.json()
        # Проверяем наличие всех обязательных полей
        assert "prediction" in data
        assert "label" in data
        assert "probabilities" in data
        assert "confidence" in data
        assert "inference_time_ms" in data
        assert "method" in data
        # Метка должна быть safe или vulnerable
        assert data["label"] in ("safe", "vulnerable"), \
            f"Unexpected label: {data['label']}"


def test_predict_endpoint_bert_method(client):
    """
    Тест: POST /predict с методом 'bert'.
    
    Проверяем, что сервис поддерживает альтернативный метод
    предсказания на основе только DistilBERT.
    """
    # Безопасный код (malloc + free)
    code = "void func() { int* p = malloc(10); free(p); }"
    response = client.post("/predict", json={"code": code, "method": "bert"})
    
    # Статус может быть 200 или 503
    assert response.status_code in (200, 503), \
        f"Unexpected status code: {response.status_code}"


def test_metrics_endpoint(client):
    """
    Тест: GET /metrics возвращает 200 и Prometheus-формат.
    
    Проверяем, что:
    - Статус 200
    - Content-Type: text/plain
    - Тело содержит ключевые метрики (REQUEST_COUNT, MODEL_STATUS)
    """
    response = client.get("/metrics")

    assert response.status_code == 200, \
        "Metrics endpoint should return 200"
    assert "text/plain" in response.headers.get("content-type", ""), \
        "Metrics should be text/plain"

    body = response.text
    assert "vulnerability_predict_requests_total" in body, \
        "Metrics should contain REQUEST_COUNT"
    assert "vulnerability_model_loaded" in body, \
        "Metrics should contain MODEL_STATUS"
    assert "vulnerability_predict_latency_seconds" in body, \
        "Metrics should contain REQUEST_LATENCY"
