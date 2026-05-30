# =============================================================================
# Модульные тесты для функций загрузки и обработки данных
# =============================================================================
# Тесты проверяют:
#   1. Дедупликацию данных (clean_data)
#   2. Фильтрацию по длине кода (clean_data)
#   3. Корректность разделения на train/val/test (prepare_splits)
#   4. Работу TF-IDF векторизации (create_tfidf_features)
#   5. Работу LSTM токенизации и паддинга (create_lstm_sequences)
# =============================================================================

import sys  # Для добавления пути к корню проекта
import os  # Работа с путями
import pytest  # Фреймворк для тестирования
import pandas as pd  # Работа с таблицами
import numpy as np  # Численные расчеты

# Добавляем корень проекта в путь импорта, чтобы находить модули src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Импортируем тестируемые функции из нашего проекта
from src.data_loader import (
    clean_data,  # Очистка данных от дубликатов и фильтрация
    prepare_splits,  # Разделение на train/val/test
    create_tfidf_features,  # TF-IDF векторизация
    create_lstm_sequences,  # LSTM токенизация + паддинг
)


@pytest.fixture
def sample_data():
    """
    Фикстура: создает синтетический тестовый датасет.
    
    Содержит 8 примеров (достаточно для стратифицированного сплита):
    4 безопасных, 4 уязвимых, разной длины.
    
    Возвращает:
        pd.DataFrame: тестовый датасет с колонками 'code' и 'label'
    """
    return pd.DataFrame({
        "code": [
            "int main() { return 0; }",  # Безопасный код (базовый)
            "void func() { char buf[10]; strcpy(buf, input); }",  # Уязвимый (strcpy)
            "void safe1() { int x = 1; }",  # Безопасный (короткий)
            "void vuln1() { char b[5]; gets(b); }",  # Уязвимый (gets)
            "int add(int a, int b) { return a + b; }",  # Безопасный (функция)
            "void leak() { int* p = malloc(10); }",  # Уязвимый (утечка памяти)
            "void loop() { for(int i=0;i<10;i++) {} }",  # Безопасный (цикл)
            "void overflow() { char d[2]; strcat(d, x); }",  # Уязвимый (strcat)
        ],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_clean_data_removes_duplicates(sample_data):
    """
    Тест: clean_data удаляет дубликаты кода.
    
    Сценарий: добавляем дубликат первого примера,
    проверяем, что clean_data удаляет его.
    """
    # Создаем датасет с дубликатом (первая строка продублирована)
    df_dup = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
    # Очищаем от дубликатов
    df_clean = clean_data(df_dup)
    # Проверяем: количество строк должно уменьшиться
    assert len(df_clean) < len(df_dup), \
        "clean_data should remove duplicate rows"


def test_clean_data_filters_by_length(sample_data):
    """
    Тест: clean_data фильтрует по длине кода (20-2000 символов).
    
    Сценарий: применяем фильтр, проверяем, что все оставшиеся
    примеры имеют длину от 20 до 2000 символов.
    """
    df_clean = clean_data(sample_data, min_len=20, max_len=2000)
    # Проверяем: результат не больше исходного
    assert len(df_clean) <= len(sample_data), \
        "clean_data should not increase dataset size"
    # Проверяем: каждый пример соответствует диапазону длин
    for code in df_clean["code"]:
        assert 20 <= len(code) <= 2000, \
            f"Code length {len(code)} should be between 20 and 2000"


def test_prepare_splits_maintains_proportions(sample_data):
    """
    Тест: prepare_splits корректно разделяет данные на 3 части.
    
    Проверяем:
    - Сумма всех частей равна исходному размеру
    - Обучающая выборка больше валидационной
    - Обе выборки не пустые
    """
    df_clean = clean_data(sample_data)
    # Разделяем на train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df_clean)

    # Сумма всех частей должна равняться исходному размеру
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(df_clean), \
        f"Split sizes should sum to {len(df_clean)}, got {total}"
    # Обучающая выборка должна быть больше валидационной
    assert len(X_train) > len(X_val), \
        "Training set should be larger than validation set"
    # Все выборки должны быть не пусты
    assert len(X_val) > 0, "Validation set should not be empty"
    assert len(X_test) > 0, "Test set should not be empty"


def test_create_tfidf_features(sample_data):
    """
    Тест: create_tfidf_features создает корректные TF-IDF признаки.
    
    Проверяем:
    - Количество строк в признаках соответствует размеру выборки
    - Векторизатор обучен (содержит vocabulary_)
    """
    df_clean = clean_data(sample_data)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df_clean)

    # Создаем TF-IDF признаки
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vec = create_tfidf_features(
        X_train, X_val, X_test
    )

    # Проверяем размерности
    assert X_train_tfidf.shape[0] == len(X_train), \
        "Train TF-IDF rows should match train size"
    assert X_test_tfidf.shape[0] == len(X_test), \
        "Test TF-IDF rows should match test size"
    # Проверяем, что векторизатор обучен
    assert hasattr(vec, "vocabulary_"), \
        "Vectorizer should be fitted (have vocabulary_)"


def test_create_lstm_sequences(sample_data):
    """
    Тест: create_lstm_sequences корректно токенизирует и паддит.
    
    Проверяем:
    - Размерность после паддинга: (n, 200)
    - Словарь содержит хотя бы 2 символа
    """
    df_clean = clean_data(sample_data)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df_clean)

    # Создаем LSTM последовательности
    X_train_pad, X_val_pad, X_test_pad, tokenizer, vocab_size = \
        create_lstm_sequences(X_train, X_val, X_test)

    # Проверяем размерность: (количество, 200)
    assert X_train_pad.shape[0] == len(X_train), \
        "Padded train rows should match train size"
    assert X_train_pad.shape[1] == 200, \
        "Padded sequences should have length 200"
    # Словарь должен содержать хотя бы 2 символа (один — padding)
    assert vocab_size > 1, \
        "Vocabulary should contain more than just padding token"
