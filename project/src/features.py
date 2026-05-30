# =============================================================================
# Модуль извлечения признаков из кода для моделей глубокого обучения
# =============================================================================
# Этот модуль предоставляет функции для:
#   1. Загрузки DistilBERT токенизатора и модели (однократное кэширование)
#   2. Извлечения [CLS]-эмбеддингов из DistilBERT (768-d вектора)
#   3. Fine-tuning DistilBERT для бинарной классификации (опционально)
#   4. Извлечения признаков из предобученной LSTM (32-d вектора)
#   5. Объединения признаков LSTM и BERT в гибридное представление (800-d)
# =============================================================================

import numpy as np  # Численные расчеты, работа с массивами
import torch  # Фреймворк PyTorch для работы с трансформерами
from tqdm import tqdm  # Прогресс-бар для длительных операций
from transformers import DistilBertTokenizer, DistilBertModel  # Трансформер DistilBERT
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Паддинг последовательностей
from tensorflow.keras.preprocessing.text import Tokenizer  # Токенизация текста


def load_bert_models(model_name="distilbert-base-uncased"):
    """
    Загрузка BERT токенизатора и модели (с кэшированием).
    
    DistilBERT — это «легкая» версия BERT, обученная методом дистилляции.
    В 2 раза быстрее и на 40% меньше оригинального BERT, сохраняя 95% качества.
    
    Функция загружает модель ОДИН раз, после чего ее можно передавать
    параметром в extract_bert_embeddings для многократного использования.
    Это критически важно, так каждая загрузка BERT с HuggingFace занимает ~5-10 сек.
    
    Параметры:
        model_name (str): название предобученной модели из HuggingFace
                         (по умолчанию 'distilbert-base-uncased')
    
    Возвращает:
        tuple: (tokenizer, model)
            - tokenizer: DistilBertTokenizer (WordPiece токенизация)
            - model: DistilBertModel (в режиме eval, без dropout)
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
    model.eval()  # Переключаем в режим инференса (отключаем dropout)
    return tokenizer, model


def extract_bert_embeddings(texts, model_name="distilbert-base-uncased",
                            batch_size=32, max_length=128,
                            tokenizer=None, model=None,
                            pooling="mean", normalize=True):
    """
    Извлечение эмбеддингов из предобученного DistilBERT.
    
    DistilBERT — это «легкая» версия BERT, обученная методом дистилляции.
    Мы используем его в режиме feature extraction: без дообучения (fine-tuning),
    просто пропускаем код через модель и забираем эмбеддинги.
    
    Поддерживаемые стратегии пулинга:
    - 'mean': среднее арифметическое всех токенов (лучше для кода)
    - 'cls': [CLS]-токен (первый токен последовательности)
    
    L2-нормализация улучшает работу LogisticRegression на эмбеддингах.
    
    Если передан tokenizer и model (из load_bert_models), они используются
    повторно, что значительно ускоряет обработку нескольких выборок.
    
    Параметры:
        texts (np.array): массив строк с исходным кодом
        model_name (str): название модели из HuggingFace (используется,
                         если не передан готовый tokenizer/model)
        batch_size (int): размер батча для обработки (32 — оптимум CPU/GPU)
        max_length (int): максимальная длина последовательности в токенах BERT
        tokenizer (DistilBertTokenizer, optional): предзагруженный токенизатор
        model (DistilBertModel, optional): предзагруженная модель
        pooling (str): стратегия пулинга ('mean' или 'cls')
        normalize (bool): L2-нормализация эмбеддингов
    
    Возвращает:
        np.array: матрица эмбеддингов размером (n_texts, 768)
    """
    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    if model is None:
        model = DistilBertModel.from_pretrained(model_name)
        model.eval()

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings"):
        batch = texts[i : i + batch_size].tolist()

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)

        if pooling == "mean":
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        else:
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        if normalize:
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

        embeddings.append(emb)

    return np.vstack(embeddings)


def finetune_bert_classifier(
    X_train, y_train, X_val, y_val,
    model_name="distilbert-base-uncased",
    max_length=128, batch_size=16, epochs=2, lr=2e-5
):
    """
    Fine-tuning DistilBERT для бинарной классификации кода.
    
    В отличие от feature extraction, fine-tuning обновляет веса BERT
    под конкретную задачу, что дает более высокое качество.
    
    Используется библиотека transformers (Trainer API).
    Из-за ограниченных ресурсов — всего 2 эпохи с маленьким learning rate.
    
    Параметры:
        X_train (list): обучающие тексты (исходный код на C/C++)
        y_train (np.array): метки обучающей выборки (0 или 1)
        X_val (list): валидационные тексты
        y_val (np.array): метки валидации
        model_name (str): предобученная модель из HuggingFace
        max_length (int): максимальная длина токенов (для обрезки)
        batch_size (int): размер батча (8-16 для CPU, 32-64 для GPU)
        epochs (int): число эпох дообучения
        lr (float): learning rate (2e-5 стандарт для fine-tuning BERT)
    
    Возвращает:
        transformers.Trainer: обученный тренер (содержит model, args, history)
    """
    from transformers import (
        DistilBertForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset  # HuggingFace Datasets для удобной работы

    # Загружаем токенизатор DistilBERT
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Функция токенизации для map()
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], padding=True, truncation=True, max_length=max_length
        )

    # Создаем HuggingFace Dataset из списков
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})
    # Токенизируем датасеты (batched=True — обрабатываем батчами)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # Определяем устройство: GPU если доступен, иначе CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Загружаем DistilBERT с головой для классификации (2 класса)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # Настройки обучения (TrainingArguments)
    # В новых версиях transformers evaluation_strategy переименован в eval_strategy
    training_kwargs = dict(
        output_dir="artifacts/models/bert-finetuned",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        disable_tqdm=False,
        dataloader_pin_memory=False,
    )
    # Поддержка обеих версий transformers (eval_strategy — новая, evaluation_strategy — старая)
    try:
        args = TrainingArguments(eval_strategy="epoch", **training_kwargs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **training_kwargs)

    # Создаем Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Запускаем обучение
    trainer.train()
    return trainer


def extract_lstm_features(model, X_pad):
    """
    Извлечение признаков из предпоследнего слоя LSTM.
    
    Берем обученную LSTM-модель, отрезаем последний (выходной) слой
    и используем выход «предпоследнего» (penultimate) слоя как
    вектор признаков размерности 32.
    
    Этот подход называется «фича-экстракшн» — мы используем нейросеть
    как преобразователь входных данных в компактное представление.
    
    Параметры:
        model (keras.Model): обученная LSTM-модель
        X_pad (np.array): паддингованные последовательности кода (n, 200)
    
    Возвращает:
        tuple: (features, feature_extractor)
            - features: матрица признаков (n_samples, 32)
            - feature_extractor: модель-экстрактор для повторного использования
    """
    from tensorflow.keras.models import Model as KerasModel

    # Создаем новую модель, которая отдает выход слоя 'penultimate'
    # (слой Dense с 32 нейронами и активацией ReLU)
    feature_extractor = KerasModel(
        inputs=model.layers[0].input,  # Вход — тот же, что у исходной модели
        outputs=model.get_layer("penultimate").output,  # Выход — предпоследний слой
    )

    # Извлекаем признаки (без обучения)
    features = feature_extractor.predict(X_pad)
    return features, feature_extractor


def build_hybrid_features(lstm_feat, bert_feat):
    """
    Объединение признаков LSTM и BERT в гибридное представление.
    
    Идея гибридного подхода:
    - LSTM захватывает локальные символьные паттерны (синтаксис кода)
    - BERT захватывает глобальный контекст (семантику кода)
    - Конкатенация дает более полное представление (800-мерный вектор)
    
    Параметры:
        lstm_feat (np.array): признаки LSTM размером (n, 32)
        bert_feat (np.array): признаки BERT размером (n, 768)
    
    Возвращает:
        np.array: объединенные признаки размером (n, 800)
    
    Исключения:
        ValueError: если количество строк в lstm_feat и bert_feat не совпадает
    """
    # Проверяем, что количество примеров совпадает
    if lstm_feat.shape[0] != bert_feat.shape[0]:
        raise ValueError(
            f"LSTM features ({lstm_feat.shape[0]}) and BERT features "
            f"({bert_feat.shape[0]}) must have same number of samples"
        )
    # Конкатенация по горизонтали (по столбцам): 32 + 768 = 800
    return np.hstack([lstm_feat, bert_feat])
