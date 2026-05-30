# =============================================================================
# Тесты для модуля извлечения признаков (src/features.py)
# =============================================================================
# Проверяют:
#   1. Корректность конкатенации LSTM и BERT признаков (build_hybrid_features)
#   2. Обработку ошибки при несовпадении размеров
#   3. Загрузку DistilBERT модели с HuggingFace
# =============================================================================

import sys  # Для добавления пути к корню проекта
import os  # Работа с путями
import pytest  # Фреймворк для тестирования
import numpy as np  # Численные расчеты

# Добавляем корень проекта в путь импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import build_hybrid_features, load_bert_models


def test_build_hybrid_features_shapes():
    """
    Тест: build_hybrid_features корректно конкатенирует признаки.
    
    Вход: LSTM (10, 32) + BERT (10, 768)
    Ожидаемый выход: (10, 800)
    """
    lstm_feat = np.random.rand(10, 32)  # 10 примеров, 32 признака
    bert_feat = np.random.rand(10, 768)  # 10 примеров, 768 признаков
    hybrid = build_hybrid_features(lstm_feat, bert_feat)
    assert hybrid.shape == (10, 800), \
        f"Expected (10, 800), got {hybrid.shape}"


def test_build_hybrid_features_mismatch():
    """
    Тест: build_hybrid_features вызывает ошибку при несовпадении размеров.
    
    Вход: LSTM (10, 32) + BERT (5, 768) → разные количества примеров
    Ожидаемый результат: ValueError
    """
    lstm_feat = np.random.rand(10, 32)  # 10 примеров
    bert_feat = np.random.rand(5, 768)  # 5 примеров (не совпадает)
    with pytest.raises(ValueError):
        build_hybrid_features(lstm_feat, bert_feat)


def test_load_bert_models():
    """
    Тест: load_bert_models загружает DistilBERT с HuggingFace.
    
    Проверяем:
    - Токенизатор не None
    - Модель не None
    - Модель имеет метод eval (режим инференса)
    """
    tokenizer, model = load_bert_models("distilbert-base-uncased")
    assert tokenizer is not None, "Tokenizer should be loaded"
    assert model is not None, "Model should be loaded"
    assert hasattr(model, "eval"), "Model should be in eval mode"
