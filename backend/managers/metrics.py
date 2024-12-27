import json
import os
import logging
from typing import Dict, Any

from fastapi import HTTPException
from pydantic import BaseModel
from typing_extensions import Annotated

from config import MODEL_DIR

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MetricsModel(BaseModel):
    metrics: Annotated[Dict[str, Any], "Метрики модели в формате JSON"]


def save_metrics(metrics: MetricsModel, model_name: Annotated[str, "Название модели"]):
    """
    Сохраняет метрики модели в файл JSON

    :param metrics: Метрики модели в формате MetricsModel
    :param model_name: Название модели
    :raises HTTPException: Если возникает ошибка при сохранении метрик
    """
    model_dir = os.path.join(MODEL_DIR, model_name)
    metrics_path = os.path.join(model_dir, "metrics.json")

    # Создание директории, если её нет
    os.makedirs(model_dir, exist_ok=True)

    try:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics.dict(), f, ensure_ascii=False, indent=4)
        logger.info(f"Метрики для модели {model_name} успешно сохранены в {metrics_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении метрик для модели {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении метрик: {e}")


def load_metrics(model_name: Annotated[str, "Название модели"]) -> MetricsModel:
    """
    Загружает метрики модели из файла JSON

    :param model_name: Название модели
    :return: Метрики модели в формате MetricsModel
    :raises HTTPException: Если файл метрик не найден или произошла другая ошибка
    """
    metrics_path = os.path.join(MODEL_DIR, model_name, "metrics.json")
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
            logger.info(f"Метрики для модели {model_name} успешно загружены из {metrics_path}")
            return MetricsModel(metrics=metrics_data)
    except FileNotFoundError:
        logger.warning(f"Метрики для модели {model_name} не найдены в {metrics_path}")
        raise HTTPException(status_code=404, detail=f"Метрики для модели {model_name} не найдены.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке метрик для модели {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке метрик: {e}")
