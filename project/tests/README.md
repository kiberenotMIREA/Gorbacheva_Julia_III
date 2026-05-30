# Тесты проекта (`tests/`)

18 автоматических тестов для ключевых модулей:

- `test_data.py` (5) — загрузка, очистка, сплиты, TF-IDF, LSTM
- `test_features.py` (3) — гибридные признаки, BERT
- `test_predict.py` (4) — инференс без моделей (bert/ensemble)
- `test_service.py` (6) — FastAPI (health, predict, валидация, metrics)

Запуск: `pytest tests -v`
