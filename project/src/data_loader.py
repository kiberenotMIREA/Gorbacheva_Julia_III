# =============================================================================
# Модуль загрузки и предобработки данных для проекта Vulnerability Scoring
# =============================================================================
# Этот модуль отвечает за:
#   1. Загрузку датасета VulDeePecker из HuggingFace Datasets
#   2. Очистку данных (дедупликация, фильтрация по длине кода)
#   3. Разделение на обучающую, валидационную и тестовую выборки
#   4. Создание TF-IDF признаков для классических ML-моделей
#   5. Подготовку последовательностей для LSTM (токенизация + паддинг)
# =============================================================================

import os  # Работа с путями файловой системы
import pickle  # Сериализация/десериализация Python-объектов
import pandas as pd  # Работа с табличными данными
import numpy as np  # Численные расчеты
from datasets import load_dataset  # Загрузка датасетов с HuggingFace
from sklearn.model_selection import train_test_split  # Разделение выборки
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF векторизация
from tensorflow.keras.preprocessing.text import Tokenizer  # Токенизация текста (символьная)
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Выравнивание последовательностей


# Константа seed для воспроизводимости результатов
# При одном и том же seed случайные разбиения дают одинаковый результат
SEED = 42


def load_vuldeepecker(
    sample_size=3000,
    test_size=500,
    random_state=SEED,
    cache_dir="data/raw",
):
    """
    Загрузка датасета VulDeePecker с HuggingFace (с локальным кэшированием).
    
    VulDeePecker содержит функции C/C++ из opensource-проектов,
    размеченные на уязвимые (1) и безопасные (0) по CWE-119 и CWE-399.
    
    Процесс загрузки:
    1. Проверяем наличие кэша в data/raw/vuldeepecker_raw.pkl
    2. Если кэш есть — загружаем из него (не качаем с HuggingFace)
    3. Если кэша нет — загружаем с HuggingFace, сохраняем в кэш
    4. Сэмплируем небольшое количество для CPU-дружественности
    5. Объединяем все в один DataFrame для сквозной обработки
    
    Параметры:
        sample_size (int): количество примеров для объединенной train+val выборки
        test_size (int): количество примеров для тестовой выборки
        random_state (int): seed для воспроизводимости сэмплирования
        cache_dir (str): директория для кэширования сырого датасета
    
    Возвращает:
        pd.DataFrame: объединенный датасет с колонками code, label, type и др.
    """
    cache_path = os.path.join(cache_dir, "vuldeepecker_raw.pkl")

    # Проверяем, есть ли кэшированная версия (чтобы не качать с HuggingFace каждый раз)
    if os.path.exists(cache_path):
        print(f"    Загрузка кэшированного датасета из {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Кэша нет — загружаем с HuggingFace
    print("    Загрузка VulDeePecker с HuggingFace...")
    ds = load_dataset("claudios/VulDeePecker")
    df_train = ds["train"].to_pandas()  # ~128k примеров (обучение)
    df_val = ds["validation"].to_pandas()  # ~16k примеров (валидация)
    df_test = ds["test"].to_pandas()  # ~16k примеров (тест)

    # Объединяем train и validation для стратифицированного разделения позже
    # Это дает больше контроля над сплитами, чем предопределенные
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)
    # Сэмплируем небольшое количество для CPU-дружественности
    df_train_val = df_train_val.sample(
        n=min(sample_size, len(df_train_val)), random_state=random_state
    )
    # Сэмплируем тестовую выборку
    df_test_small = df_test.sample(
        n=min(test_size, len(df_test)), random_state=random_state
    )

    # Переименовываем колонку для единообразия (в исходных данных она называется functionSource)
    df_train_val = df_train_val.rename(columns={"functionSource": "code"})
    df_test_small = df_test_small.rename(columns={"functionSource": "code"})
    # Добавляем текстовую метку для удобства визуализации
    df_train_val["type"] = df_train_val["label"].map({0: "safe", 1: "vulnerable"})
    df_test_small["type"] = df_test_small["label"].map({0: "safe", 1: "vulnerable"})

    # Объединяем все в один DataFrame для сквозной обработки
    df = pd.concat([df_train_val, df_test_small], ignore_index=True)

    # Сохраняем в локальный кэш (data/raw/) — не попадет в git (игнорируется .gitignore)
    print(f"    Сохранение кэшированного датасета в {cache_path}")
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)

    return df


def clean_data(df, min_len=20, max_len=2000):
    """
    Очистка данных от дубликатов и фильтрация по длине кода.
    
    Этапы очистки:
    1. Дедупликация — удаление одинаковых функций (они могут встречаться
       в разных CWE-категориях, что искажает распределение)
    2. Фильтрация по длине — удаление слишком коротких (<20 символов)
       и слишком длинных (>2000 символов) фрагментов кода
    
    Слишком короткие фрагменты неинформативны (например, "int x;"),
    слишком длинные требуют больших вычислительных ресурсов и часто
    содержат шум.
    
    Параметры:
        df (pd.DataFrame): исходный датасет с колонкой 'code'
        min_len (int): минимальная длина кода в символах
        max_len (int): максимальная длина кода в символах
    
    Возвращает:
        pd.DataFrame: очищенный датасет
    """
    # Шаг 1: удаляем дубликаты по содержимому кода
    df_unique = df.drop_duplicates(subset=["code"])

    # Шаг 2: фильтруем по длине — слишком короткие фрагменты неинформативны,
    # слишком длинные требуют больших вычислительных ресурсов
    df_filtered = df_unique[
        (df_unique["code"].apply(len) >= min_len)
        & (df_unique["code"].apply(len) <= max_len)
    ]
    return df_filtered


def prepare_splits(df, test_size=0.15, val_ratio=0.176, random_state=SEED):
    """
    Стратифицированное разделение данных на train/val/test.
    
    Схема разделения:
    - 70% — обучающая выборка (train)
    - 15% — валидационная выборка (val) — для подбора гиперпараметров
    - 15% — тестовая выборка (test) — для финальной оценки
    
    val_ratio=0.176 означает, что от оставшихся после выделения test 85% данных
    мы берем 17.6%, что составляет ~15% от общего объема.
    
    Стратификация сохраняет пропорцию классов (vulnerable/safe) во всех выборках.
    Это критически важно при сильном дисбалансе (~5.7% уязвимых).
    
    Параметры:
        df (pd.DataFrame): очищенный датасет с колонками 'code' и 'label'
        test_size (float): доля тестовой выборки (0.15 = 15%)
        val_ratio (float): доля валидационной выборки от train (0.176)
        random_state (int): seed для воспроизводимости
    
    Возвращает:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
            - X_*: numpy массивы строк с кодом
            - y_*: numpy массивы меток (0 или 1)
    """
    # Извлекаем признаки (код) и целевую переменную (метка уязвимости)
    X = df["code"].to_numpy(dtype=str, na_value="")
    y = df["label"].to_numpy(dtype=np.int64)

    # Шаг 1: отделяем тестовую выборку (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Шаг 2: из оставшихся 85% отделяем валидационную (17.6% от 85% = 15% от общего)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_tfidf_features(X_train, X_val, X_test, max_features=10000, ngram_range=(3, 6)):
    """
    Создание TF-IDF признаков на уровне символов (char-level).
    
    TF-IDF (Term Frequency — Inverse Document Frequency) преобразует текст
    в числовую матрицу, где каждый столбец соответствует n-грамме символов.
    
    Используются символьные n-граммы (3-6 символов), так как код C/C++
    имеет характерные синтаксические паттерны (например, "strcpy", "malloc").
    
    Параметры:
        X_train (np.array): обучающие тексты кода
        X_val (np.array): валидационные тексты кода
        X_test (np.array): тестовые тексты кода
        max_features (int): максимальное количество признаков (n-грамм)
        ngram_range (tuple): диапазон длин n-грамм (мин, макс)
    
    Возвращает:
        tuple: (X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer)
            - X_*_tfidf: разреженные матрицы TF-IDF признаков
            - vectorizer: обученный TfidfVectorizer (для сохранения)
    """
    # Создаем векторизатор: char-level, n-граммы 3-6 символов
    vec = TfidfVectorizer(
        analyzer="char", ngram_range=ngram_range, max_features=max_features
    )

    # fit_transform — обучаем словарь и преобразуем обучающую выборку
    X_train_tfidf = vec.fit_transform(X_train)
    # transform — только преобразуем (словарь уже обучен на train)
    X_val_tfidf = vec.transform(X_val)
    X_test_tfidf = vec.transform(X_test)

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vec


def create_lstm_sequences(X_train, X_val, X_test, maxlen=200):
    """
    Подготовка последовательностей для LSTM-модели.
    
    Процесс:
    1. Символьная токенизация — каждый символ кода преобразуется в число
    2. Паддинг — все последовательности приводятся к единой длине (maxlen)
    
    Символьная токенизация выбрана, потому что:
    - Словарный запас символов C/C++ ограничен (~70-100 символов)
    - Позволяет модели изучать синтаксические паттерны на уровне символов
    - Не требует сложной предобработки текста
    
    Параметры:
        X_train (np.array): обучающие тексты кода
        X_val (np.array): валидационные тексты кода
        X_test (np.array): тестовые тексты кода
        maxlen (int): максимальная длина последовательности (в символах)
                     Тексты длиннее обрезаются, короче — дополняются нулями
    
    Возвращает:
        tuple: (X_train_pad, X_val_pad, X_test_pad, tokenizer, vocab_size)
            - X_*_pad: паддингованные последовательности (n, maxlen)
            - tokenizer: обученный Tokenizer (для сохранения и инференса)
            - vocab_size: размер словаря (количество уникальных символов + 1)
    """
    # Создаем токенизатор на уровне символов
    # oov_token — токен для неизвестных символов (out-of-vocabulary)
    tokenizer = Tokenizer(char_level=True, oov_token="<UNK>")

    # Обучаем токенизатор на обучающей выборке (строим словарь символов)
    tokenizer.fit_on_texts(X_train)

    # Преобразуем тексты в последовательности чисел (каждому символу — свой индекс)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Паддинг: обрезаем/дополняем все последовательности до maxlen
    # pre padding (добавляем нули в начало) — стандарт для LSTM
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

    # Размер словаря = количество уникальных символов + 1 (для padding)
    vocab_size = len(tokenizer.word_index) + 1

    return X_train_pad, X_val_pad, X_test_pad, tokenizer, vocab_size


def save_processed_data(df, filepath="data/processed/dataset_project.pkl"):
    """
    Сохранение обработанного датасета в pickle-файл.
    
    Параметры:
        df (pd.DataFrame): датасет для сохранения
        filepath (str): путь к файлу сохранения
    """
    # Создаем директорию, если ее нет
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(df, f)


def load_processed_data(filepath="data/processed/dataset_project.pkl"):
    """
    Загрузка ранее сохраненного обработанного датасета.
    
    Параметры:
        filepath (str): путь к файлу с данными
    
    Возвращает:
        pd.DataFrame: загруженный датасет
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)
