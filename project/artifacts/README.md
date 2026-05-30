# Артефакты проекта (`artifacts/`)

- `models/` — обученные модели (.keras, .pkl, .json)
- `figures/` — графики и ROC-кривые
- `logs/` — логи обучения (TensorBoard)

Генерируются автоматически при запуске `python -m src.train`.
Очистка: `scripts/cleanup.bat` (Windows) или `scripts/cleanup.sh` (Linux).
