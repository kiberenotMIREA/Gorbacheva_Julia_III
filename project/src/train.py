# =============================================================================
# Пайплайн обучения всех моделей для проекта Vulnerability Scoring
# =============================================================================
# Этот скрипт реализует полный цикл обучения и сравнения моделей:
#   1. Загрузка датасета VulDeePecker с HuggingFace
#   2. Очистка данных (дедупликация, фильтрация по длине)
#   3. Разделение на train/val/test (70/15/15)
#   4. Baseline 1: TF-IDF + LogisticRegression
#   5. Нейросеть: Bidirectional LSTM (с EarlyStopping)
#   6. Трансформер: DistilBERT + LogisticRegression (feature extraction)
#   7. DistilBERT fine-tuning (опционально)
#   8. Ансамбль Stacking: BERT-LR + BERT-embed + LSTM → мета-классификатор
#   9. Сравнение и выбор лучшей модели по F1-score
# =============================================================================

import os
import sys
import json
import argparse

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import yaml
import mlflow  # Трекинг экспериментов

# Метрики из scikit-learn для оценки качества классификации
from sklearn.metrics import (
    classification_report,  # Полный отчет: precision, recall, f1 по классам
    confusion_matrix,  # Матрица ошибок (TP, FP, FN, TN)
    roc_auc_score,  # Площадь под ROC-кривой
    precision_score,  # Точность: TP / (TP + FP)
    recall_score,  # Полнота: TP / (TP + FN)
    f1_score,  # Гармоническое среднее precision и recall
    accuracy_score,  # Доля правильных ответов
)
from sklearn.linear_model import LogisticRegression  # Для мета-классификатора
import joblib  # Сохранение/загрузка scikit-learn моделей

# Импортируем модули нашего проекта
from src.data_loader import (
    load_vuldeepecker,  # Загрузка датасета с HuggingFace
    clean_data,  # Очистка: дедупликация + фильтрация по длине
    prepare_splits,  # Стратифицированное разделение на train/val/test
    create_tfidf_features,  # Char-level TF-IDF векторизация
    create_lstm_sequences,  # Символьная токенизация + паддинг для LSTM
    save_processed_data,  # Сохранение обработанного датасета в pickle
)
from src.features import (
    extract_bert_embeddings,  # Извлечение [CLS]-эмбеддингов DistilBERT
    load_bert_models,  # Загрузка BERT токенизатора и модели (однократно)
    build_hybrid_features,  # Конкатенация LSTM (32-d) + BERT (768-d)
    finetune_bert_classifier,  # Fine-tuning DistilBERT для классификации
)
from src.models import (
    build_lstm_model,  # Создание архитектуры BiLSTM
    compute_class_weights,  # Расчет весов классов для дисбаланса
    train_logistic_regression,  # Обучение LogisticRegression

    train_hybrid_classifier,  # Обучение мета-классификатора ансамбля
    train_meta_classifier,  # Универсальное обучение мета-классификатора
    save_model,  # Универсальное сохранение (Keras .keras / sklearn .pkl)
)

# Константы по умолчанию (переопределяются из config.yaml)
SEED = 42  # Фиксированный seed для воспроизводимости результатов
MODELS_DIR = "artifacts/models"  # Директория для сохранения обученных моделей
DATA_DIR = "data/processed"  # Директория для обработанных данных


def load_config(config_path="configs/config.yaml"):
    """
    Загрузка конфигурации из YAML-файла.
    
    Параметры:
        config_path (str): путь к файлу конфигурации
    
    Возвращает:
        dict: словарь с настройками данных, предобработки, моделей и сервиса
    
    Пример структуры config.yaml:
        data:
          sample_size: 3000
          test_size: 500
          random_seed: 42
        preprocessing:
          test_split: 0.15
          maxlen: 200
        models:
          lstm:
            epochs: 20
            batch_size: 32
          bert:
            model_name: distilbert-base-uncased
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_best_threshold(y_true, y_proba):
    """
    Поиск оптимального порога решающего правила по F1-score.
    
    Стандартный порог 0.5 не всегда оптимален при дисбалансе классов.
    Эта функция перебирает пороги от 0.05 до 0.95 и находит тот,
    при котором F1 на валидационной выборке максимален.
    
    Параметры:
        y_true (np.array): истинные метки (0 или 1)
        y_proba (np.array): предсказанные вероятности класса 1
    
    Возвращает:
        tuple: (best_threshold, best_f1) — оптимальный порог и достигнутый F1
    """
    # Перебираем 91 значение порога с шагом 0.01
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1 = 0
    best_t = 0.5  # По умолчанию — стандартный порог
    for t in thresholds:
        # Бинаризуем вероятности по текущему порогу
        y_pred = (y_proba >= t).astype(int)
        # Считаем F1 (zero_division=0 — не падаем, если нет предсказаний класса)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t, best_f1


def evaluate(y_test, y_pred, y_proba):
    """
    Вычисление набора метрик для бинарной классификации.
    
    Параметры:
        y_test (np.array): истинные метки тестовой выборки
        y_pred (np.array): предсказанные метки (0 или 1)
        y_proba (np.array): предсказанные вероятности класса 1
    
    Возвращает:
        dict: словарь с метриками accuracy, precision, recall, f1, roc_auc
    """
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_pipeline(config_path="configs/config.yaml"):
    """
    Главная функция пайплайна обучения.

    Выполняет полный цикл: от загрузки данных до сохранения всех моделей
    и вывода сравнительной таблицы. В конце определяет лучшую модель по F1.

    Параметры:
        config_path (str): путь к YAML-файлу конфигурации

    Возвращает:
        dict: словарь с метриками всех обученных моделей
    """
    # Загружаем конфигурацию из YAML
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    preproc_cfg = cfg["preprocessing"]
    lstm_cfg = cfg["models"]["lstm"]
    bert_cfg = cfg["models"]["bert"]
    # Устанавливаем seed из конфига и создаем директории
    global SEED, MODELS_DIR, DATA_DIR
    SEED = data_cfg.get("random_seed", 42)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Настройка MLflow для трекинга экспериментов
    tracking_cfg = cfg.get("tracking", {})
    tracking_uri = os.getenv("TRACKING_URI", tracking_cfg.get("tracking_uri", "sqlite:///artifacts/mlruns/mlflow.db"))
    experiment_name = tracking_cfg.get("experiment_name", "vulnerability-scoring")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name="training-pipeline", nested=False)

    # Логируем версию конфига и параметры данных
    mlflow.log_params({
        "sample_size": data_cfg["sample_size"],
        "test_size": data_cfg["test_size"],
        "seed": SEED,
        "min_code_len": data_cfg.get("min_code_len", 20),
        "max_code_len": data_cfg.get("max_code_len", 2000),
    })
    mlflow.log_param("config_path", os.path.abspath(config_path))
    # Логируем гиперпараметры всех моделей
    mlflow.log_params({
        "lstm_embedding_dim": lstm_cfg.get("embedding_dim", 64),
        "lstm_units_1": lstm_cfg.get("lstm_units_1", 64),
        "lstm_units_2": lstm_cfg.get("lstm_units_2", 32),
        "lstm_dropout": lstm_cfg.get("dropout", 0.5),
        "lstm_epochs": lstm_cfg.get("epochs", 20),
        "lstm_batch_size": lstm_cfg.get("batch_size", 32),
        "lstm_learning_rate": lstm_cfg.get("learning_rate", 0.0005),
        "lstm_max_class_weight": lstm_cfg.get("max_class_weight", 3.0),
        "bert_model_name": bert_cfg.get("model_name", "distilbert-base-uncased"),
        "bert_max_length": bert_cfg.get("max_length", 128),
        "bert_batch_size": bert_cfg.get("batch_size", 32),
        "lr_max_iter": cfg["models"]["logistic_regression"].get("max_iter", 1000),
        "tfidf_max_features": preproc_cfg.get("tfidf_max_features", 10000),
        "tfidf_ngram_range": str(preproc_cfg.get("tfidf_ngram_range", [3, 6])),
        "maxlen": preproc_cfg.get("maxlen", 200),
        "test_split": preproc_cfg.get("test_split", 0.15),
    })

    print("=" * 60)
    print("VULNERABILITY SCORING — ПАЙПЛАЙН ОБУЧЕНИЯ")
    print("=" * 60)

    # =========================================================================
    # Шаг 1: Загрузка датасета
    # =========================================================================
    # Загружаем подвыборку VulDeePecker с HuggingFace.
    # Параметры sample_size и test_size задаются в config.yaml.
    print("\n[1] Загрузка датасета VulDeePecker...")
    df = load_vuldeepecker(
        sample_size=data_cfg["sample_size"],
        test_size=data_cfg["test_size"],
        random_state=SEED,
    )
    # Очищаем: удаляем дубликаты и слишком короткие/длинные фрагменты
    df_filtered = clean_data(
        df,
        min_len=data_cfg.get("min_code_len", 20),
        max_len=data_cfg.get("max_code_len", 2000),
    )
    # Сохраняем очищенный датасет для повторного использования
    save_processed_data(df_filtered, f"{DATA_DIR}/dataset_project.pkl")
    print(f"    Очищенных данных: {len(df_filtered)} образцов")

    # =========================================================================
    # Шаг 2: Разделение на train/val/test
    # =========================================================================
    # Стратифицированное разделение: сохраняем пропорцию классов в каждой выборке.
    # 70% — обучение, 15% — валидация, 15% — тест.
    print("\n[2] Разделение на train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(
        df_filtered,
        test_size=preproc_cfg.get("test_split", 0.15),
        val_ratio=preproc_cfg.get("val_split_from_train", 0.176),
        random_state=SEED,
    )
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Словарь для хранения метрик всех моделей (ключ — имя модели, значение — dict метрик)
    results = {}

    # =========================================================================
    # Шаг 3: Baseline 1 — TF-IDF + LogisticRegression
    # =========================================================================
    # Простая линейная модель на char-level TF-IDF признаках.
    # Служит нижней границей качества (baseline).
    print("\n[3] Baseline 1: TF-IDF + LogisticRegression...")
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vec = create_tfidf_features(
        X_train, X_val, X_test,
        max_features=preproc_cfg.get("tfidf_max_features", 10000),
        ngram_range=tuple(preproc_cfg.get("tfidf_ngram_range", [3, 6])),
    )
    lr = train_logistic_regression(
        X_train_tfidf, y_train,
        max_iter=cfg["models"]["logistic_regression"].get("max_iter", 1000),
    )
    y_pred_lr = lr.predict(X_test_tfidf)
    y_proba_lr = lr.predict_proba(X_test_tfidf)[:, 1]
    results["LogisticRegression"] = evaluate(y_test, y_pred_lr, y_proba_lr)
    print(f"    F1: {results['LogisticRegression']['f1']:.3f}, "
          f"Recall: {results['LogisticRegression']['recall']:.3f}")
    save_model(vec, f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    save_model(lr, f"{MODELS_DIR}/lr_classifier.pkl")

    # =========================================================================
    # Шаг 4: Нейросеть — Bidirectional LSTM
    # =========================================================================
    # Двунаправленная LSTM для анализа последовательностей символов кода.
    # Используем class weights для учета дисбаланса классов и
    # EarlyStopping + ReduceLROnPlateau для предотвращения переобучения.
    print("\n[4] Нейросеть: Bidirectional LSTM...")
    X_train_lstm, X_val_lstm, X_test_lstm, tokenizer, vocab_size = \
        create_lstm_sequences(
            X_train, X_val, X_test,
            maxlen=preproc_cfg.get("maxlen", 200),
        )
    lstm_model = build_lstm_model(
        vocab_size,
        maxlen=preproc_cfg.get("maxlen", 200),
        embedding_dim=lstm_cfg.get("embedding_dim", 64),
        lstm_units_1=lstm_cfg.get("lstm_units_1", 64),
        lstm_units_2=lstm_cfg.get("lstm_units_2", 32),
        dropout=lstm_cfg.get("dropout", 0.5),
        dense_units=lstm_cfg.get("dense_units", 32),
        learning_rate=lstm_cfg.get("learning_rate", 0.0005),
    )
    # Веса классов: меньший класс (уязвимый) получает больший вес
    class_weights = compute_class_weights(
        y_train, max_weight=lstm_cfg.get("max_class_weight", 3.0)
    )
    print(f"    Веса классов: {class_weights}")

    # Колбэки для управления обучением
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    callbacks = [
        EarlyStopping(
            monitor="val_loss",  # Следим за loss на валидации
            patience=3,  # Если 3 эпохи без улучшения — останавливаем
            restore_best_weights=True,  # Восстанавливаем лучшие веса
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,  # Уменьшаем learning rate в 2 раза
            patience=2,  # Если 2 эпохи без улучшения
            min_lr=1e-6,  # Минимальный LR
            verbose=1,
        ),
    ]

    lstm_model.fit(
        X_train_lstm, y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=lstm_cfg.get("epochs", 20),
        batch_size=lstm_cfg.get("batch_size", 32),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Оптимальный порог по валидации, затем предсказание на тесте
    y_proba_lstm = lstm_model.predict(X_test_lstm).flatten()
    lstm_thresh, _ = find_best_threshold(y_val, lstm_model.predict(X_val_lstm).flatten())
    y_pred_lstm = (y_proba_lstm >= lstm_thresh).astype(int)
    results["BiLSTM"] = evaluate(y_test, y_pred_lstm, y_proba_lstm)
    print(f"    F1: {results['BiLSTM']['f1']:.3f}, "
          f"Recall: {results['BiLSTM']['recall']:.3f}")
    save_model(lstm_model, f"{MODELS_DIR}/lstm_model.keras")
    # Сохраняем токенизатор в JSON для использования в сервисе
    with open(f"{MODELS_DIR}/tokenizer.json", "w") as f:
        json.dump(tokenizer.to_json(), f)
    # Создаём и сохраняем экстрактор LSTM-признаков (penultimate layer, 32-d)
    from tensorflow.keras.models import Model as KerasModel
    feat_extractor = KerasModel(
        inputs=lstm_model.layers[0].input,
        outputs=lstm_model.get_layer("penultimate").output,
    )
    feat_extractor.save(f"{MODELS_DIR}/lstm_feature_extractor.keras")
    print(f"    Экстрактор признаков сохранён: {MODELS_DIR}/lstm_feature_extractor.keras")
    # Извлекаем penultimate признаки (32-d) для Ensemble
    lstm_train_feat = feat_extractor.predict(X_train_lstm, verbose=0)
    lstm_val_feat = feat_extractor.predict(X_val_lstm, verbose=0)
    lstm_test_feat = feat_extractor.predict(X_test_lstm, verbose=0)
    print(f"    LSTM признаки: train {lstm_train_feat.shape}, "
          f"val {lstm_val_feat.shape}, test {lstm_test_feat.shape}")

    # =========================================================================
    # Шаг 5: Трансформер — DistilBERT (feature extraction)
    # =========================================================================
    # Используем DistilBERT в режиме извлечения признаков (без fine-tuning).
    # Загружаем модель ОДИН раз и передаем во все вызовы extract_bert_embeddings.
    # Это дает 768-мерные [CLS]-эмбеддинги для каждого фрагмента кода.
    print("\n[5] Трансформер: DistilBERT + LogisticRegression...")
    bert_tokenizer, bert_model = load_bert_models(
        model_name=bert_cfg.get("model_name", "distilbert-base-uncased")
    )
    bert_train_feat = extract_bert_embeddings(
        X_train, tokenizer=bert_tokenizer, model=bert_model,
        batch_size=bert_cfg.get("batch_size", 32),
        max_length=bert_cfg.get("max_length", 128),
    )
    bert_val_feat = extract_bert_embeddings(
        X_val, tokenizer=bert_tokenizer, model=bert_model,
        batch_size=bert_cfg.get("batch_size", 32),
        max_length=bert_cfg.get("max_length", 128),
    )
    bert_test_feat = extract_bert_embeddings(
        X_test, tokenizer=bert_tokenizer, model=bert_model,
        batch_size=bert_cfg.get("batch_size", 32),
        max_length=bert_cfg.get("max_length", 128),
    )
    # LogisticRegression на BERT-эмбеддингах с учетом дисбаланса
    bert_clf = train_logistic_regression(
        bert_train_feat, y_train,
        max_iter=cfg["models"]["logistic_regression"].get("max_iter", 1000),
    )
    y_proba_bert = bert_clf.predict_proba(bert_test_feat)[:, 1]
    bert_thresh, _ = find_best_threshold(y_val, bert_clf.predict_proba(bert_val_feat)[:, 1])
    y_pred_bert = (y_proba_bert >= bert_thresh).astype(int)
    results["DistilBERT+LR"] = evaluate(y_test, y_pred_bert, y_proba_bert)
    print(f"    F1: {results['DistilBERT+LR']['f1']:.3f}, "
          f"Recall: {results['DistilBERT+LR']['recall']:.3f}")
    save_model(bert_clf, f"{MODELS_DIR}/bert_classifier.pkl")

    # =========================================================================
    # Шаг 5b: DistilBERT fine-tuning (опционально)
    # =========================================================================
    # Дополнительное улучшение: дообучаем DistilBERT на нашей задаче.
    # Из-за ограниченных ресурсов — только 2 эпохи, batch_size=8.
    # Если fine-tuning не удался (нет GPU, мало памяти) — пропускаем.
    ft_epochs = bert_cfg.get("finetune_epochs", 0)
    if ft_epochs > 0:
        print(f"\n[6b] DistilBERT Fine-Tuning (опционально, {ft_epochs} эпохи)...")
        try:
            ft_trainer = finetune_bert_classifier(
                X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train),
                y_train,
                X_val.tolist() if hasattr(X_val, 'tolist') else list(X_val),
                y_val,
                model_name=bert_cfg.get("model_name", "distilbert-base-uncased"),
                max_length=bert_cfg.get("max_length", 128),
                batch_size=8,
                epochs=ft_epochs,
            )
            ft_trainer.save_model(f"{MODELS_DIR}/bert-finetuned")
            print("    Дообученный BERT сохранен.")
        except Exception as e:
            print(f"    Fine-tuning пропущен ({e})")

    # =========================================================================
    # Шаг 6: Ансамбль Stacking (BERT + LSTM)
    # =========================================================================
    #   Мета-признаки:
    #      - BERT-LR proba (1-d)
    #      - BERT mean embedding (768-d, напрямую)
    #      - LSTM penultimate признаки (32-d)
    #   3. Сравнение мета-классификаторов (LR, GB, SGD)
    #   4. Выбор лучшего по F1-score на тестовой выборке
    #   1. Мета-признаки:
    #      - BERT_LR proba (1-d, single split)
    #      - BERT mean embedding (768-d, напрямую, не только через LR)
    #      - LSTM penultimate features (32-d, отдельно от BERT)
    #   2. Сравнение нескольких мета-классификаторов (LR, GB, SGD)
    print("\n[6] Ансамбль: Stacking (BERT + LSTM)...")

    # --- Мета-признаки для train (single split для BERT/LSTM) ---
    bert_proba_train_1d = bert_clf.predict_proba(bert_train_feat)[:, 1].reshape(-1, 1)

    X_meta_train = np.hstack([
        bert_proba_train_1d,     # 1-d (BERT->LR вероятность)
        bert_train_feat,         # 768-d (BERT mean embedding, напрямую)
        lstm_train_feat,         # 32-d (LSTM penultimate признаки)
    ])

    # --- Мета-признаки для val ---
    bert_proba_val_1d = bert_clf.predict_proba(bert_val_feat)[:, 1].reshape(-1, 1)

    X_meta_val = np.hstack([
        bert_proba_val_1d,
        bert_val_feat,
        lstm_val_feat,
    ])

    # --- Мета-признаки для test ---
    bert_proba_test_1d = bert_clf.predict_proba(bert_test_feat)[:, 1].reshape(-1, 1)

    X_meta_test = np.hstack([
        bert_proba_test_1d,
        bert_test_feat,
        lstm_test_feat,
    ])

    print(f"    Мета-признаки: {X_meta_train.shape[1]}d "
          f"(BERT_LR + BERT_embed + LSTM)")

    # --- Сравнение мета-классификаторов ---
    ensemble_candidates = cfg["models"].get("ensemble", {}).get("candidates", [])
    if not ensemble_candidates:
        ensemble_candidates = [{
            "meta_type": "logistic_regression",
            "scaler_type": "standard",
            "label": "Ensemble_LR+Std",
            "params": {},
        }]

    best_ensemble_f1 = 0
    best_ensemble_key = None
    best_ensemble_clf = None
    best_ensemble_scaler = None
    best_ensemble_thresh = 0.5

    for cand in ensemble_candidates:
        label = cand["label"]
        print(f"\n  [{label}] {cand['meta_type']} + {cand['scaler_type']}...")
        try:
            params = dict(cand.get("params", {}))
            params.setdefault("max_iter", cfg["models"]["logistic_regression"].get("max_iter", 1000))
            clf, scaler = train_meta_classifier(
                X_meta_train, y_train,
                meta_type=cand["meta_type"],
                scaler_type=cand["scaler_type"],
                **params,
            )

            X_meta_val_scaled = scaler.transform(X_meta_val) if scaler else X_meta_val
            X_meta_test_scaled = scaler.transform(X_meta_test) if scaler else X_meta_test

            if hasattr(clf, "predict_proba"):
                y_proba_meta = clf.predict_proba(X_meta_test_scaled)[:, 1]
                y_proba_val_m = clf.predict_proba(X_meta_val_scaled)[:, 1]
            else:
                y_proba_meta = clf.decision_function(X_meta_test_scaled)
                y_proba_val_m = clf.decision_function(X_meta_val_scaled)

            meta_thresh, _ = find_best_threshold(y_val, y_proba_val_m)
            y_pred_meta = (y_proba_meta >= meta_thresh).astype(int)
            results[label] = evaluate(y_test, y_pred_meta, y_proba_meta)
            print(f"    F1: {results[label]['f1']:.3f}, "
                  f"Recall: {results[label]['recall']:.3f}")

            f1 = results[label]["f1"]
            # Предпочитаем LR при равном F1 (LR стабильнее градиентного бустинга)
            is_lr = "LR" in label
            is_best_lr = "LR" in (best_ensemble_key or "")
            margin = 0.01  # Не-LR должен быть на 0.01 лучше, чтобы выиграть
            if is_lr:
                if best_ensemble_key is None or f1 >= best_ensemble_f1 - margin:
                    best_ensemble_f1 = f1
                    best_ensemble_key = label
                    best_ensemble_clf = clf
                    best_ensemble_scaler = scaler
                    best_ensemble_thresh = meta_thresh
            else:
                if best_ensemble_key is None or f1 > best_ensemble_f1 + margin:
                    best_ensemble_f1 = f1
                    best_ensemble_key = label
                    best_ensemble_clf = clf
                    best_ensemble_scaler = scaler
                    best_ensemble_thresh = meta_thresh

        except Exception as e:
            print(f"    Ошибка: {e}, пропускаем")

    if best_ensemble_clf is not None:
        save_model(best_ensemble_clf, f"{MODELS_DIR}/ensemble_meta_clf.pkl")
        save_model(best_ensemble_scaler, f"{MODELS_DIR}/ensemble_meta_scaler.pkl")
        # Сохраняем оптимальный порог для корректного инференса
        np.save(f"{MODELS_DIR}/ensemble_meta_thresh.npy", np.array([best_ensemble_thresh]))
        print(f"\n  Лучший ансамбль: {best_ensemble_key} "
              f"(F1={best_ensemble_f1:.3f}, threshold={meta_thresh:.3f})")

    # =========================================================================
    # Шаг 7: Вывод итогов и определение лучшей модели
    # =========================================================================
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО — СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    summary = summary.round(3)
    print(summary)

    # Сохраняем сводную таблицу в CSV для анализа
    summary.to_csv(f"{DATA_DIR}/model_comparison.csv")
    print(f"\nРезультаты сохранены в {DATA_DIR}/model_comparison.csv")

    # Лучшая модель — по максимальному F1-score
    best_model = max(results, key=lambda k: results[k]["f1"])
    print(f"\nЛучшая модель: {best_model} (F1={results[best_model]['f1']:.3f})")

    # Логируем все метрики в MLflow
    mlflow.log_param("best_model", best_model)
    for model_name, metrics in results.items():
        safe_name = model_name.replace("+", "_").replace(" ", "_")
        mlflow.log_metrics(
            {f"{safe_name}/{k}": v for k, v in metrics.items()},
        )
    mlflow.log_artifact(f"{DATA_DIR}/model_comparison.csv")
    mlflow.log_artifact(config_path)
    mlflow.log_artifacts(MODELS_DIR, artifact_path="models")
    mlflow.log_param("run_id", run.info.run_id)
    print(f"\nMLflow run: {run.info.run_id}")
    print(f"MLflow UI: mlflow ui --backend-store-uri {tracking_uri}")

    mlflow.end_run()
    return results


# Точка входа: запуск пайплайна с поддержкой аргументов командной строки
if __name__ == "__main__":
    # argparse позволяет указать путь к конфигу:
    #   python -m src.train --config configs/my_config.yaml
    parser = argparse.ArgumentParser(
        description="Vulnerability Scoring — пайплайн обучения"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Путь к YAML-файлу конфигурации"
    )
    args = parser.parse_args()
    train_pipeline(config_path=args.config)
