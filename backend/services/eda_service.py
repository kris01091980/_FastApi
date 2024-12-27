import base64
import io
import logging

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_plots(file):
    """
    Генерирует графики на основе загруженного CSV файла.

    file: файл в формате CSV, загружаемый пользователем
    Возвращает словарь с графиками в формате base64
    """
    plots = {}

    try:
        # Загружаем данные
        df = pd.read_csv(io.StringIO(str(file.file.read(), 'utf-8')))
        logger.info("Данные успешно загружены")
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return {"error": f"Ошибка загрузки данных: {e}"}

    # Выбираем только числовые столбцы
    numeric_columns = df.select_dtypes(include=["float", "int"]).columns
    if numeric_columns.empty:
        logger.warning("Нет числовых столбцов для анализа")
        return {"plots": {}, "message": "Нет числовых столбцов для анализа"}

    # График распределения для первого числового столбца
    try:
        if len(numeric_columns) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(
                df['trans_month'],
                kde=True,
                ax=ax,
                color="skyblue",
                edgecolor="black",
                linewidth=1.5,
            )
            ax.set_title(
                f"Распределение: trans_month",
                fontsize=16,
                fontweight="bold",
                color="darkblue",
            )
            ax.set_xlabel(
                'Номер месяца',
                fontsize=14,
                fontweight="bold",
            )
            ax.set_ylabel(
                "Частота",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, linestyle="--", alpha=0.7)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plots["distribution"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()
            logger.info(f"График распределения для {numeric_columns[0]} успешно создан")
    except Exception as e:
        logger.error(f"Ошибка при создании графика распределения: {e}")
        return plots

    # Матрица корреляции
    try:
        if len(numeric_columns) > 1:
            relevant_columns = [col for col in numeric_columns if not col.lower().startswith("unnamed")]
            if relevant_columns:
                fig, ax = plt.subplots(figsize=(14, 10))
                corr = df[relevant_columns].corr()
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    annot_kws={"size": 10, "fontweight": "bold"},
                    cbar_kws={"shrink": 0.8, "aspect": 30},
                    linewidths=0.5,
                    square=True,
                    ax=ax,
                )
                ax.set_title(
                    "Матрица корреляции",
                    fontsize=18,
                    fontweight="bold",
                    color="darkblue",
                    pad=20,
                )
                ax.tick_params(axis="x", labelsize=12, rotation=45)
                ax.tick_params(axis="y", labelsize=12)
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                plots["correlation_matrix"] = base64.b64encode(buf.read()).decode("utf-8")
                plt.close()
                logger.info("Матрица корреляции успешно создана")
    except Exception as e:
        logger.error(f"Ошибка при создании матрицы корреляции: {e}")
        return plots

    return plots
