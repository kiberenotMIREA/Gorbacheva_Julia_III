# Исходный код проекта (`src/`)

Модули проекта, разделенные по функциональности:

- `data_loader.py` — загрузка VulDeePecker, очистка, сплиты, TF-IDF, LSTM-токенизация
- `features.py` — извлечение BERT-эмбеддингов, LSTM-признаков, гибридных признаков
- `models.py` — архитектуры (BiLSTM), обучение (LR, Ensemble)
- `train.py` — полный пайплайн: 4 модели, сравнение, сохранение
- `predict.py` — класс `VulnerabilityPredictor` (инференс bert/ensemble)
- `service.py` — FastAPI сервис (/predict, /health, /methods)

Запуск: `python -m src.train` (обучение) или `python -m src.service` (API)
