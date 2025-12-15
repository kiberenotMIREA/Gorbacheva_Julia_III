from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    #Ограничение анализа категорий
    max_columns_top_categories: int = typer.Option(
        5,
        help="Максимальное количество категориальных колонок для анализа топ-значений."
    ),
    #Кастомный заголовок
    title: str = typer.Option(
        "EDA-отчёт",
        help="Заголовок отчёта в Markdown файле."
    ),
    #Контроль детализации категорий
    top_k_categories: int = typer.Option(
        5,
        help="Сколько топ-значений выводить для категориальных признаков."
    ),
    #Чувствительность к пропускам
    min_missing_share: float = typer.Option(
        0.3,
        help="Порог доли пропусков для выделения проблемных колонок (от 0.0 до 1.0)."
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    #Валидация параметров
    if max_columns_top_categories <= 0:
        raise typer.BadParameter("max-columns-top-categories должен быть положительным числом")
    if top_k_categories <= 0:
        raise typer.BadParameter("top-k-categories должен быть положительным числом")
    if min_missing_share < 0 or min_missing_share > 1:
        raise typer.BadParameter("min-missing-share должен быть между 0.0 и 1.0")
    
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    # Используем новые параметры при вызове функций
    top_cats = top_categories(
        df, 
        max_columns=max_columns_top_categories,  # Передаём max_columns из параметра CLI
        top_k=top_k_categories  # Передаём top_k из параметра CLI
    )

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df)

    # Дополнительный анализ: проблемные колонки по пропускам
    problematic_missing_cols = []
    if not missing_df.empty:
        problematic_missing_cols = missing_df[missing_df["missing_share"] > min_missing_share]
    
    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")  # Используем параметр title
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        # Добавляем информацию о параметрах анализа
        f.write("## Параметры анализа\n\n")
        f.write(f"- Макс. гистограмм: **{max_hist_columns}**\n")
        f.write(f"- Топ-категорий для вывода: **{top_k_categories}**\n")
        f.write(f"- Макс. категориальных колонок для анализа: **{max_columns_top_categories}**\n")
        f.write(f"- Порог пропусков для проблемных колонок: **{min_missing_share:.0%}**\n\n")
    
        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n\n")
        
        # Добавляем новые эвристики из compute_quality_flags
        if 'has_constant_columns' in quality_flags:
            f.write(f"- Есть константные колонки: **{quality_flags['has_constant_columns']}**\n")
            if quality_flags['has_constant_columns'] and quality_flags['constant_columns']:
                f.write(f"  - Константные колонки: {', '.join(quality_flags['constant_columns'])}\n")
        
        if 'has_high_cardinality_categoricals' in quality_flags:
            f.write(f"- Есть категориальные с высокой кардинальностью: **{quality_flags['has_high_cardinality_categoricals']}**\n")
            if quality_flags['has_high_cardinality_categoricals'] and quality_flags['high_cardinality_columns']:
                for col_info in quality_flags['high_cardinality_columns']:
                    f.write(f"  - `{col_info['column']}`: {col_info['unique_values']} уникальных значений\n")
        
        if 'has_suspicious_id_duplicates' in quality_flags:
            f.write(f"- Есть подозрительные дубликаты ID: **{quality_flags['has_suspicious_id_duplicates']}**\n")
        
        f.write("\n")
        
        # Добавляем раздел с проблемными колонками (по пропускам)
        if not problematic_missing_cols.empty:
            f.write("##Проблемные колонки (много пропусков)\n\n")
            f.write(f"Колонки с долей пропусков > {min_missing_share:.0%}:\n\n")
            for col_name, row in problematic_missing_cols.iterrows():
                f.write(f"- **{col_name}**: {row['missing_count']} пропусков ({row['missing_share']:.1%})\n")
            f.write("\n")
        elif not missing_df.empty:
            f.write("## Пропуски\n\n")
            f.write(f"Все колонки имеют долю пропусков ≤ {min_missing_share:.0%}.\n\n")
       
        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"Анализировано колонок: {len(top_cats)} (первые {max_columns_top_categories})\n")
            f.write(f"Топ-значений на колонку: {top_k_categories}\n\n")
            f.write("Подробные таблицы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        numeric_cols = [c.name for c in summary.columns if c.is_numeric]
        f.write(f"Всего числовых колонок: {len(numeric_cols)}\n")
        if len(numeric_cols) > max_hist_columns:
            f.write(f"Создано гистограмм: {max_hist_columns} (первые {max_hist_columns} колонок)\n")
            f.write(f"Не показаны: {', '.join(numeric_cols[max_hist_columns:])}\n")
        else:
            f.write(f"Создано гистограмм: {len(numeric_cols)}\n")
        f.write("\nФайлы: `hist_*.png`\n")

        # Добавляем примечание об использованных параметрах
        f.write("\n---\n")
        f.write("*Отчёт создан с параметрами:*\n")
        f.write(f"- `--max-hist-columns={max_hist_columns}`\n")
        f.write(f"- `--top-k-categories={top_k_categories}`\n")
        f.write(f"- `--title=\"{title}\"`\n")
        f.write(f"- `--min-missing-share={min_missing_share}`\n")
        f.write(f"- `--max-columns-top-categories={max_columns_top_categories}`\n")

    # 5. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")
    
    # Выводим использованные параметры
    typer.echo("\nИспользованные параметры:")
    typer.echo(f"Заголовок: {title}")
    typer.echo(f"Макс. гистограмм: {max_hist_columns}")
    typer.echo(f"Топ-категорий: {top_k_categories}")
    typer.echo(f"Порог пропусков: {min_missing_share:.0%}")
    typer.echo(f"Макс. категориальных колонок: {max_columns_top_categories}")

if __name__ == "__main__":
    app()
