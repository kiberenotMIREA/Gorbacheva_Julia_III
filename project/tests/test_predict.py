# =============================================================================
# Тесты для модуля инференса (src/predict.py)
# =============================================================================
# Проверяют:
#   1. Инициализацию предсказателя без обученных моделей
#   2. Обработку неизвестного метода предсказания
#   3. Graceful degradation при отсутствии моделей (bert, ensemble)
# =============================================================================

import sys  # Для добавления пути к корню проекта
import os  # Работа с путями
import pytest  # Фреймворк для тестирования

# Добавляем корень проекта в путь импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import VulnerabilityPredictor


def test_predictor_no_models():
    """
    Тест: предсказатель инициализируется без обученных моделей.
    
    Проверяем, что:
    - Предсказатель создается (не падает)
    - BERT модель загружается из HuggingFace (всегда доступна)
    - Ensemble классификатор отсутствует (файла нет)
    """
    predictor = VulnerabilityPredictor(models_dir="nonexistent_path_xyz")
    assert predictor is not None, "Predictor should initialize"
    # BERT всегда загружается из HuggingFace (требует интернет)
    assert "bert_model" in predictor.components
    assert "bert_tokenizer" in predictor.components
    # Ensemble классификатор не загружен (файла нет)
    assert "meta_clf" not in predictor.components


def test_predict_unknown_method():
    """
    Тест: неизвестный метод предсказания возвращает (None, None).
    """
    predictor = VulnerabilityPredictor(models_dir="nonexistent_path_xyz")
    pred, proba = predictor.predict("int main() { return 0; }", method="invalid")
    assert pred is None, "Prediction should be None for unknown method"
    assert proba is None, "Probabilities should be None for unknown method"


def test_predict_bert_no_models():
    """
    Тест: bert-метод без моделей возвращает (None, None).
    Даже если BERT загружен, классификатор может отсутствовать.
    """
    predictor = VulnerabilityPredictor(models_dir="nonexistent_path_xyz")
    pred, proba = predictor.predict_bert("int main() { return 0; }")
    assert pred is None
    assert proba is None


def test_predict_ensemble_no_models():
    """
    Тест: ensemble-метод без моделей возвращает (None, None).
    """
    predictor = VulnerabilityPredictor(models_dir="nonexistent_path_xyz")
    pred, proba = predictor.predict_ensemble("int main() { return 0; }")
    assert pred is None
    assert proba is None
