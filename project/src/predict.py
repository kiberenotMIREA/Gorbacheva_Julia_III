# =============================================================================
# Модуль инференса (получения предсказаний) для Vulnerability Scoring API
# =============================================================================
# Этот модуль содержит класс VulnerabilityPredictor, который:
#   1. Загружает все обученные компоненты модели с диска
#   2. Предоставляет методы для предсказания: bert, ensemble
#   3. Обрабатывает один фрагмент кода и возвращает предсказание + вероятности
#
    # Поддерживаемые методы:
    #   - 'bert':     DistilBERT эмбеддинги → LogisticRegression
    #   - 'ensemble': ансамбль Stacking (BERT + LSTM)
    # =============================================================================

import os
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import torch

import sys
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from transformers import DistilBertTokenizer, DistilBertModel


# Путь к директории с сохраненными моделями по умолчанию
MODELS_DIR = "artifacts/models"


class VulnerabilityPredictor:
    """
    Класс для предсказания уязвимостей в C/C++ коде.
    
    Инкапсулирует всю логику загрузки моделей и инференса.
    Поддерживает два метода предсказания:
    - 'bert':     DistilBERT эмбеддинги → LogisticRegression
    - 'ensemble': ансамбль Stacking (BERT + LSTM)
    
    Атрибуты:
        models_dir (str): путь к директории с моделями
        components (dict): словарь загруженных компонентов:
            - tokenizer: символьный токенизатор Keras
            - feat_extractor: экстрактор LSTM-признаков (предпоследний слой)
            - bert_tokenizer: токенизатор DistilBERT (WordPiece)
            - bert_model: модель DistilBERT (режим eval)
            - bert_clf: LogisticRegression на BERT-эмбеддингах
            - meta_clf: мета-классификатор ансамбля
    """
    
    def __init__(self, models_dir=MODELS_DIR):
        """
        Инициализация предсказателя.
        
        Параметры:
            models_dir (str): путь к директории с сохраненными моделями
        """
        self.models_dir = models_dir
        self.components = {}  # Словарь для хранения загруженных компонентов
        self._load_components()  # Загружаем все доступные компоненты

    def _load_components(self):
        """
        Загрузка всех компонентов модели с диска.
        
        Загружает только те компоненты, файлы которых существуют.
        Это позволяет сервису работать даже если не все модели обучены.
        
        Загружаемые компоненты:
        - Токенизатор LSTM (tokenizer.json)
        - Экстрактор LSTM-признаков (lstm_feature_extractor.keras)
        - Гибридный классификатор (hybrid_classifier.pkl) и scaler (hybrid_scaler.pkl)
        - BERT-классификатор (bert_classifier.pkl)
        - Мета-классификатор ансамбля (ensemble_meta_clf.pkl) и scaler
        - DistilBERT токенизатор и модель (из HuggingFace, требуется интернет)
        """
        feat_path = f"{self.models_dir}/lstm_feature_extractor.keras"
        tok_path = f"{self.models_dir}/tokenizer.json"
        bert_clf_path = f"{self.models_dir}/bert_classifier.pkl"

        vec_path = f"{self.models_dir}/tfidf_vectorizer.pkl"
        meta_clf_path = f"{self.models_dir}/ensemble_meta_clf.pkl"

        # Загружаем токенизатор LSTM (символьный, char-level)
        if os.path.exists(tok_path):
            with open(tok_path, "r") as f:
                self.components["tokenizer"] = tokenizer_from_json(json.load(f))

        # Загружаем экстрактор LSTM-признаков (предпоследний слой BiLSTM)
        # Используется только экстрактор, полная LSTM модель не загружается
        if os.path.exists(feat_path):
            self.components["feat_extractor"] = load_model(feat_path, compile=False)

        if os.path.exists(bert_clf_path):
            self.components["bert_clf"] = joblib.load(bert_clf_path)
        if os.path.exists(vec_path):
            self.components["vec"] = joblib.load(vec_path)
        if os.path.exists(meta_clf_path):
            self.components["meta_clf"] = joblib.load(meta_clf_path)
        meta_scaler_path = f"{self.models_dir}/ensemble_meta_scaler.pkl"
        if os.path.exists(meta_scaler_path):
            self.components["meta_scaler"] = joblib.load(meta_scaler_path)
        thresh_path = f"{self.models_dir}/ensemble_meta_thresh.npy"
        if os.path.exists(thresh_path):
            self.components["meta_thresh"] = float(np.load(thresh_path)[0])

        # Загружаем DistilBERT: сначала пробуем локальную fine-tuned копию,
        # затем — из HuggingFace (кэш/интернет).
        # Оборачиваем в try-except из-за возможных проблем с памятью или сетью.
        bert_local_path = f"{self.models_dir}/bert-finetuned"
        bert_source = "distilbert-base-uncased"
        if os.path.isdir(bert_local_path):
            bert_source = bert_local_path

        try:
            self.components["bert_tokenizer"] = \
                DistilBertTokenizer.from_pretrained(bert_source)
            self.components["bert_model"] = \
                DistilBertModel.from_pretrained(bert_source)
            self.components["bert_model"].eval()  # Режим инференса (без dropout)
        except Exception as e:
            print(f"Предупреждение: DistilBERT не загружен ({e}). "
                  f"Ансамбль может работать некорректно.")

    def _get_bert_embedding(self, text, pooling="mean"):
        """
        Извлечение эмбеддинга DistilBERT для одного текста.

        Использует mean pooling (усреднение по всем токенам) вместо [CLS],
        что обычно дает более робастное представление для кода.
        L2-нормализация улучшает работу LogisticRegression.

        Параметры:
            text (str): фрагмент кода C/C++ для анализа
            pooling (str): стратегия пулинга ('mean' или 'cls')

        Возвращает:
            np.array: эмбеддинг размерности (1, 768) или None, если BERT не загружен
        """
        tokenizer = self.components.get("bert_tokenizer")
        model = self.components.get("bert_model")
        if tokenizer is None or model is None:
            return None

        inputs = tokenizer(
            [text], padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)

        if pooling == "mean":
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        else:
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # L2-нормализация
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return emb

    def _get_lstm_features(self, text):
        """
        Извлечение LSTM-признаков (32-d) для одного текста.
        
        Использует экстрактор признаков из предпоследнего слоя BiLSTM.
        Экстрактор преобразует последовательность символов кода
        в компактный 32-мерный вектор, выделяя локальные паттерны.
        
        Параметры:
            text (str): фрагмент кода C/C++
        
        Возвращает:
            np.array: признаки размерности (1, 32) или None, если LSTM не загружен
        """
        tokenizer = self.components.get("tokenizer")
        feat_extractor = self.components.get("feat_extractor")
        if tokenizer is None or feat_extractor is None:
            return None

        # Токенизация и паддинг до 200 символов (как при обучении)
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=200)
        # Извлекаем признаки (verbose=0 — без вывода прогресс-бара)
        return feat_extractor.predict(pad, verbose=0)

    def predict_bert(self, text):
        """
        Предсказание через DistilBERT + LogisticRegression.
        
        Использует только BERT-эмбеддинги (без LSTM).
        Быстрее, чем гибридный метод, но обычно менее точный.
        
        Параметры:
            text (str): фрагмент кода C/C++
        
        Возвращает:
            tuple: (prediction, probabilities) или (None, None) при ошибке
        """
        clf = self.components.get("bert_clf")
        if clf is None:
            return None, None

        emb = self._get_bert_embedding(text)
        if emb is None:
            return None, None

        pred = clf.predict(emb)[0]
        proba = clf.predict_proba(emb)[0]
        return int(pred), proba.tolist()

    def predict_ensemble(self, text):
        """
        Предсказание через ансамбль Stacking.

        Ансамбль комбинирует:
        - Предсказание BERT+LR (вероятность)
        - BERT mean embedding (768-d, напрямую)
        - LSTM penultimate признаки (32-d)
        Мета-классификатор обучается на этой комбинации (801-d).

        Параметры:
            text (str): фрагмент кода C/C++

        Возвращает:
            tuple: (prediction, probabilities) или (None, None) при ошибке
        """
        clf = self.components.get("meta_clf")
        if clf is None:
            return None, None

        bert_feat = self._get_bert_embedding(text)
        lstm_feat = self._get_lstm_features(text)
        if bert_feat is None or lstm_feat is None:
            return None, None

        bert_clf = self.components.get("bert_clf")
        if bert_clf is None:
            return None, None
        bert_proba = bert_clf.predict_proba(bert_feat)[:, 1].reshape(1, -1)

        # Мета-признаки: BERT_LR(1) + BERT_embed(768) + LSTM(32)
        meta_feat = np.hstack([
            bert_proba,
            bert_feat, lstm_feat,
        ])
        scaler = self.components.get("meta_scaler")
        if scaler:
            meta_feat = scaler.transform(meta_feat)
        proba = clf.predict_proba(meta_feat)[0]
        thresh = self.components.get("meta_thresh", 0.5)
        pred = 1 if proba[1] >= thresh else 0
        return int(pred), proba.tolist()

    def predict(self, text, method="hybrid"):
        """
        Единый метод предсказания с выбором модели.
        
        Параметры:
            text (str): фрагмент кода C/C++
            method (str): метод предсказания:
                - 'bert': DistilBERT + LogisticRegression
                - 'ensemble': ансамбль Stacking (по умолчанию)
        
        Возвращает:
            tuple: (prediction, probabilities) или (None, None)
        """
        if method == "bert":
            return self.predict_bert(text)
        elif method == "ensemble":
            return self.predict_ensemble(text)
        # Неизвестный метод — возвращаем None
        return None, None
