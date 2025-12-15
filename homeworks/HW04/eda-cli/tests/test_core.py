from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )

# Тестовый датафрейм с константной колонкой
def _df_with_constant_column() -> pd.DataFrame:
    """Создаёт DataFrame с константной колонкой для тестирования."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "constant_col": [5, 5, 5, 5, 5],  # Константная колонка - все значения одинаковые
        "normal_col": [10, 20, 30, 40, 50],  # Нормальная колонка с разными значениями
        "status": ["active"] * 5,  # Ещё одна константная колонка, но строкового типа
    })

# Тестовый датафрейм с высокой кардинальностью
def _df_with_high_cardinality() -> pd.DataFrame:
    """Создаёт DataFrame с категориальной колонкой с высокой кардинальностью."""
    # Создаём 150 уникальных значений для колонки 'user_id'
    user_ids = [f"user_{i:03d}" for i in range(150)]
    return pd.DataFrame({
        "user_id": user_ids,  # Высокая кардинальность: 150 уникальных значений при 150 строках
        "category": ["A", "B", "C"] * 50,  # Низкая кардинальность: всего 3 значения
        "value": range(150)
    })

# Тестовый датафрейм с дубликатами ID
def _df_with_id_duplicates() -> pd.DataFrame:
    """Создаёт DataFrame с дублирующимися ID."""
    return pd.DataFrame({
        "customer_id": [101, 102, 101, 103, 102],  # Дубликаты: 101 и 102 повторяются
        "order_id": ["A001", "A002", "A003", "A004", "A005"],  # Уникальные ID
        "user_name": ["alice", "bob", "alice", "charlie", "bob"],
        "amount": [100, 200, 150, 300, 250]
    })

# Тестовый датафрейм с ID-подобными колонками без дубликатов
def _df_with_unique_ids() -> pd.DataFrame:
    """Создаёт DataFrame с уникальными ID."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],  # Уникальный числовой ID
        "user_id": ["U001", "U002", "U003", "U004", "U005"],  # Уникальный строковый ID
        "transaction_id": ["T001", "T002", "T003", "T004", "T005"],  # Уникальный ID
        "value": [10, 20, 30, 40, 50]
    })

def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

# НОВЫЙ ТЕСТ 1:Проверка обнаружения константных колонок
def test_constant_columns_detection():
    """Тестируем, что функция правильно находит константные колонки."""
    
    # Создаём тестовый DataFrame с константной колонкой
    df = _df_with_constant_column()
    
    # Получаем статистику и пропуски
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вычисляем флаги качества
    flags = compute_quality_flags(summary, missing_df)
    
    # Флаг должен быть True
    assert flags["has_constant_columns"] is True, \
        "Флаг has_constant_columns должен быть True при наличии константных колонок"
    
    # Должны быть найдены 2 константные колонки
    constant_cols = flags["constant_columns"]
    assert len(constant_cols) == 2, \
        f"Ожидалось 2 константные колонки, но найдено {len(constant_cols)}"
    
    # Конкретные имена константных колонок
    assert "constant_col" in constant_cols, \
        "Колонка 'constant_col' должна быть в списке константных"
    assert "status" in constant_cols, \
        "Колонка 'status' должна быть в списке константных"
    
    # Нормальная колонка не должна быть в списке
    assert "normal_col" not in constant_cols, \
        "Колонка 'normal_col' не должна быть в списке константных"
    
    # quality_score должен быть снижен из-за константных колонок
    assert flags["quality_score"] < 1.0, \
        "Score качества должен быть снижен из-за константных колонок"
    
    print(" Тест на константные колонки пройден!")
    print(f" Найдены константные колонки: {constant_cols}")
    print(f" Оценка качества: {flags['quality_score']:.2f}")

# НОВЫЙ ТЕСТ 2: Проверка обнаружения высокой кардинальности
def test_high_cardinality_detection():
    """Тестируем, что функция правильно находит категории с высокой кардинальностью."""
    
    # Создаём тестовый DataFrame с высокой кардинальностью
    df = _df_with_high_cardinality()
    
    # Получаем статистику и пропуски
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вычисляем флаги качества
    flags = compute_quality_flags(summary, missing_df)
    
    # Флаг должен быть True
    assert flags["has_high_cardinality_categoricals"] is True, \
        "Флаг has_high_cardinality_categoricals должен быть True при высокой кардинальности"
    
    # Должна быть найдена 1 колонка с высокой кардинальностью
    high_card_cols = flags["high_cardinality_columns"]
    assert len(high_card_cols) == 1, \
        f"Ожидалась 1 колонка с высокой кардинальностью, но найдено {len(high_card_cols)}"
    
    # Это должна быть колонка 'user_id'
    assert high_card_cols[0]["column"] == "user_id", \
        f"Колонка с высокой кардинальностью должна быть 'user_id', а не '{high_card_cols[0]['column']}'"
    
    # Уникальных значений должно быть 150
    assert high_card_cols[0]["unique_values"] == 150, \
        f"Ожидалось 150 уникальных значений, но найдено {high_card_cols[0]['unique_values']}"
    
    # Колонка 'category' не должна быть в списке (всего 3 значения)
    category_cols = [col["column"] for col in high_card_cols]
    assert "category" not in category_cols, \
        "Колонка 'category' не должна быть в списке высокой кардинальности"
    
    # quality_score должен быть снижен
    assert flags["quality_score"] < 1.0, \
        "Score качества должен быть снижен из-за высокой кардинальности"
    
    print(" Тест на высокую кардинальность пройден!")
    print(f" Найдена колонка: {high_card_cols[0]['column']}")
    print(f" Уникальных значений: {high_card_cols[0]['unique_values']}")
    print(f" Оценка качества: {flags['quality_score']:.2f}")

# НОВЫЙ ТЕСТ 3: Проверка обнаружения дубликатов ID
def test_id_duplicates_detection():
    """Тестируем, что функция правильно находит дубликаты в ID-подобных колонках."""
    
    # Создаём тестовый DataFrame с дубликатами ID
    df = _df_with_id_duplicates()
    
    # Получаем статистику и пропуски
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вычисляем флаги качества
    flags = compute_quality_flags(summary, missing_df)
    
    # Флаг должен быть True
    assert flags["has_suspicious_id_duplicates"] is True, \
        "Флаг has_suspicious_id_duplicates должен быть True при наличии дубликатов ID"
    
    # Должны быть найдены 2 проблемные колонки
    suspicious_cols = flags["suspicious_id_columns"]
    assert len(suspicious_cols) >= 1, \
        f"Ожидалась хотя бы 1 проблемная колонка, но найдено {len(suspicious_cols)}"
    
    # 'customer_id' должен быть в списке
    customer_id_found = any(col["column"] == "customer_id" for col in suspicious_cols)
    assert customer_id_found is True, \
        "Колонка 'customer_id' должна быть в списке подозрительных"
    
    # 'user_name' может быть в списке (содержит 'id' в названии)
    user_name_found = any(col["column"] == "user_name" for col in suspicious_cols)
    # user_name содержит 'id' в названии? Нет, поэтому не должно быть найдено
    # Но может быть найдено из-за шаблонов, если есть 'id' в названии
    
    # Проверяем детали для customer_id
    for col_info in suspicious_cols:
        if col_info["column"] == "customer_id":
            # 5 строк, 3 уникальных значения
            assert col_info["unique_values"] == 3, \
                f"customer_id должен иметь 3 уникальных значения, а имеет {col_info['unique_values']}"
            assert "duplicate_count" in col_info or "uniqueness_ratio" in col_info, \
                "Должна быть информация о дубликатах"
    
    # ✅quality_score должен быть значительно снижен
    # Штраф за дубликаты ID: 0.25
    assert flags["quality_score"] < 0.75, \
        f"Score качества должен быть значительно снижен из-за дубликатов ID. Текущий: {flags['quality_score']}"
    
    print(" Тест на дубликаты ID пройден!")
    print(f" Найдено проблемных колонок: {len(suspicious_cols)}")
    for col in suspicious_cols:
        print(f"  - {col['column']}: {col.get('duplicate_count', 'N/A')} дубликатов")

# НОВЫЙ ТЕСТ 4: Проверка что уникальные ID не считаются проблемой
def test_unique_ids_no_problem():
    """Тестируем, что уникальные ID не помечаются как проблемные."""
    
    # Создаём тестовый DataFrame с уникальными ID
    df = _df_with_unique_ids()
    
    # Получаем статистику и пропуски
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вычисляем флаги качества
    flags = compute_quality_flags(summary, missing_df)
    
    # Флаг должен быть False (нет дубликатов)
    assert flags["has_suspicious_id_duplicates"] is False, \
        "Флаг has_suspicious_id_duplicates должен быть False при уникальных ID"
    
    # Список должен быть пустым
    suspicious_cols = flags["suspicious_id_columns"]
    assert len(suspicious_cols) == 0, \
        f"Список подозрительных колонок должен быть пустым, но содержит {len(suspicious_cols)} элементов"
    
    print(" Тест на уникальные ID пройден!")
    print(f" Подозрительных колонок: {len(suspicious_cols)}")

# НОВЫЙ ТЕСТ 5: Комплексный тест со всеми проблемами
def test_all_heuristics_together():
    """Тестируем все новые эвристики на одном DataFrame."""
    
    # Создаём комплексный тестовый DataFrame
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],  # Уникальный ID (не проблема)
        "constant_flag": [1, 1, 1, 1, 1],  # Константная колонка
        "user_hash": [f"hash_{i}" for i in range(5)],  # Высокая кардинальность (5 из 5)
        "customer_id": [101, 101, 102, 103, 104],  # Дубликаты ID
        "normal_col": [10, 20, 30, 40, 50],  # Нормальная колонка
    })
    
    # Получаем статистику и пропуски
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вычисляем флаги качества
    flags = compute_quality_flags(summary, missing_df)
    
    # Константные колонки
    assert flags["has_constant_columns"] is True
    assert "constant_flag" in flags["constant_columns"]
    
    # Высокая кардинальность
    # user_hash: 5 уникальных из 5 строк = 100% - это не проблема по нашему порогу (100)
    # Порог высокой кардинальности = 100, а у нас только 5
    assert flags["has_high_cardinality_categoricals"] is False
    
    # Дубликаты ID
    assert flags["has_suspicious_id_duplicates"] is True
    customer_found = any(col["column"] == "customer_id" for col in flags["suspicious_id_columns"])
    assert customer_found is True
    
    # Score качества должен быть значительно снижен
    # Штрафы: константные колонки (-0.15) + дубликаты ID (-0.25) = -0.40
    # Базовый score: 1.0, ожидаем около 0.60 (минус другие возможные штрафы)
    assert flags["quality_score"] < 0.7, \
        f"Score качества должен быть < 0.7 из-за множества проблем. Текущий: {flags['quality_score']}"
    
    print(" Комплексный тест пройден!")
    print(f" Оценка качества: {flags['quality_score']:.2f}")
    print(f" Константные колонки: {flags['constant_columns']}")
    print(f" Дубликаты ID: {[col['column'] for col in flags['suspicious_id_columns']]}")

