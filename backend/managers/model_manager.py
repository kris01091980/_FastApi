import os
import pickle
import logging
from typing import List

import pandas as pd
from fastapi import HTTPException

from config import MODEL_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Создание директории для моделей, если она не существует
os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(model, model_name: str):
    """
    Сохраняет модель в указанной директории.

    model: объект обученной модели, который нужно сохранить
    model_name: уникальное имя модели, используемое для хранения
    Возвращает полный путь к сохранённому файлу
    """
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Модель {model_name} успешно сохранена в {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении модели: {e}")


def load_model(model_name: str):
    """
    Загружает модель из директории по её имени.

    model_name: имя модели для загрузки
    Возвращает загруженный объект модели
    Генерирует HTTPException в случае ошибки или отсутствия файла
    """
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Модель {model_name} не найдена")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Модель {model_name} успешно загружена из {model_path}")
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке модели: {e}")


def list_available_models() -> list:
    """
    Возвращает список доступных моделей, хранящихся в директории.

    Возвращает список имён файлов моделей без расширения
    Генерирует HTTPException в случае ошибки при чтении директории
    """
    try:
        model_files = [f.replace('.pkl', '') for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
        logger.info(f"Доступные модели: {model_files}")
        return model_files
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении списка моделей: {e}")


class DataPreprocessingError(Exception):
    """
    Исключение для ошибок, возникающих во время предобработки данных.

    message: описание ошибки
    """

    def __init__(self, message: str):
        super().__init__(message)


class MissingColumnsError(Exception):
    """
    Исключение для случаев, когда в данных отсутствуют обязательные столбцы.

    missing_columns: список отсутствующих столбцов
    required_columns: полный список обязательных столбцов
    """

    def __init__(self, missing_columns: List[str], required_columns: List[str]):
        self.missing_columns = missing_columns
        self.required_columns = required_columns
        message = f"Отсутствуют столбцы: {', '.join(missing_columns)}"
        super().__init__(message)


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]):
    """
    Проверяет наличие всех обязательных столбцов в DataFrame.

    df: DataFrame, который нужно проверить
    required_columns: список обязательных столбцов
    Генерирует MissingColumnsError, если не хватает необходимых столбцов
    """
    missing_columns = list(set(required_columns) - set(df.columns))
    if missing_columns:
        raise MissingColumnsError(missing_columns, required_columns)
