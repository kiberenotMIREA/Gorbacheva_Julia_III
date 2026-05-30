# =============================================================================
# Модуль определения и обучения моделей для обнаружения уязвимостей
# =============================================================================
# Этот модуль содержит функции для создания и обучения всех моделей проекта:
#   1. BiLSTM — двунаправленная рекуррентная нейросеть для анализа кода
#   2. LogisticRegression — линейная модель (baseline и финальный классификатор)
#   3. Stacking Ensemble — мета-классификатор на мета-признаках (BERT + LSTM)
# =============================================================================

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import joblib
import os

# Seed для воспроизводимости результатов (используется во всех моделях)
SEED = 42


def build_lstm_model(vocab_size, maxlen=200, embedding_dim=64,
                     lstm_units_1=64, lstm_units_2=32,
                     dropout=0.5, dense_units=32, learning_rate=0.001):
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        Bidirectional(LSTM(lstm_units_1, return_sequences=True)),
        Dropout(dropout),
        Bidirectional(LSTM(lstm_units_2)),
        Dropout(dropout),
        Dense(dense_units, activation="relu", name="penultimate"),
        Dense(1, activation="sigmoid"),
    ], name="BiLSTM")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(y_train, max_weight=3.0):
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weights = np.clip(weights, None, max_weight)
    return dict(enumerate(weights))


def train_logistic_regression(X_train, y_train, max_iter=1000, class_weight="balanced"):
    """
    Обучение логистической регрессии.
    
    Логистическая регрессия — простая линейная модель для бинарной классификации.
    Используется как baseline (на TF-IDF) и как финальный классификатор
    для гибридной модели и ансамбля.
    
    Параметр class_weight='balanced' автоматически подбирает веса
    обратно пропорционально частоте классов (аналогично compute_class_weights).
    
    Параметры:
        X_train (np.array/sparse): обучающие признаки
        y_train (np.array): метки (0 или 1)
        max_iter (int): максимальное количество итераций сходимости
        class_weight (str/dict): веса классов ('balanced' или словарь)
    
    Возвращает:
        LogisticRegression: обученная модель
    """
    lr = LogisticRegression(
        max_iter=max_iter,
        random_state=SEED,
        class_weight=class_weight,  # Автоматический учет дисбаланса
    )
    lr.fit(X_train, y_train)
    return lr


def build_meta_classifier(meta_type="logistic_regression", scaler_type="standard", **kwargs):
    """
    Создание мета-классификатора и scaler по заданному типу.

    Параметры:
        meta_type (str): тип мета-классификатора:
            - 'logistic_regression' — LogisticRegression (linear)
            - 'gradient_boosting'   — GradientBoostingClassifier (tree-based)
            - 'sgd'                 — SGDClassifier (loss='log' = logistic)
        scaler_type (str): тип scaler:
            - 'standard' — StandardScaler (z-score)
            - 'minmax'   — MinMaxScaler [0, 1]
            - 'robust'   — RobustScaler (медиана + IQR)
            - 'none'     — без масштабирования
        **kwargs: дополнительные параметры для классификатора

    Возвращает:
        tuple: (clf, scaler, needs_scaling)
            - clf: необученный классификатор
            - scaler: scaler (None если scaler_type='none')
            - needs_scaling (bool): нужна ли стандартизация перед predict
    """
    SEED = kwargs.get("random_state", 42)
    class_weight = kwargs.get("class_weight", "balanced")
    n_jobs = kwargs.get("n_jobs", -1)
    max_iter = kwargs.get("max_iter", 1000)

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = None

    if meta_type == "logistic_regression":
        clf = LogisticRegression(
            max_iter=max_iter,
            random_state=SEED,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
    elif meta_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 3),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=SEED,
        )
    elif meta_type == "sgd":
        clf = SGDClassifier(
            loss=kwargs.get("sgd_loss", "log_loss"),
            max_iter=kwargs.get("sgd_epochs", 1000),
            tol=kwargs.get("sgd_tol", 1e-3),
            random_state=SEED,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
    elif meta_type == "svm":
        clf = SVC(
            probability=kwargs.get("probability", True),
            kernel=kwargs.get("kernel", "rbf"),
            class_weight=class_weight,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Неизвестный тип мета-классификатора: {meta_type}")

    return clf, scaler, scaler is not None


def train_meta_classifier(X_meta, y, meta_type="logistic_regression",
                          scaler_type="standard", **kwargs):
    """
    Обучение мета-классификатора ансамбля.
    
    Параметры:
        X_meta (np.array): мета-признаки (n, n_features)
        y (np.array): метки (0 или 1)
        meta_type (str): тип мета-классификатора
        scaler_type (str): тип scaler
        **kwargs: дополнительные параметры

    Возвращает:
        tuple: (classifier, scaler)
            - classifier: обученный классификатор
            - scaler: scaler (None если не используется)
    """
    clf, scaler, needs_scale = build_meta_classifier(
        meta_type=meta_type, scaler_type=scaler_type, **kwargs
    )

    if needs_scale and scaler is not None:
        X_meta = scaler.fit_transform(X_meta)

    clf.fit(X_meta, y)
    return clf, scaler


def train_hybrid_classifier(X_train_hyb, y_train, max_iter=1000):
    """
    Обучение мета-классификатора (LogisticRegression + StandardScaler).
    
    Сохранено для обратной совместимости; рекомендуется использовать
    train_meta_classifier вместо direct. 
    Делегирует вызов в train_meta_classifier с параметрами по умолчанию.
    """
    return train_meta_classifier(
        X_train_hyb, y_train,
        meta_type="logistic_regression",
        scaler_type="standard",
        max_iter=max_iter,
    )


def save_model(model, path):
    """
    Сохранение модели на диск.
    
    Поддерживает два формата:
    - Keras-модели (.keras/.h5): использует model.save()
    - Scikit-learn модели (.pkl): использует joblib.dump()
    
    Параметры:
        model: модель для сохранения (keras.Model или sklearn estimator)
        path (str): путь к файлу сохранения
    """
    # Создаем директорию, если ее нет
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # У Keras-моделей есть метод .save(), у sklearn — используем joblib
    if hasattr(model, "save"):
        model.save(path)  # Keras сохраняет в .keras формат
    else:
        joblib.dump(model, path)  # Scikit-learn через joblib


def load_sklearn_model(path):
    """
    Загрузка scikit-learn модели из файла.
    
    Параметры:
        path (str): путь к файлу модели (.pkl)
    
    Возвращает:
        object: загруженная модель (LogisticRegression, scaler и т.п.)
    """
    return joblib.load(path)
