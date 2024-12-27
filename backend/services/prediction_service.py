import base64
import os
import logging

from managers.model_manager import load_model, validate_required_columns
from managers.metrics import load_metrics

from config import REQUIRED_COLUMNS, MODEL_DIR
from preprocessing import preprocess_data

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def predict(model_name: str, df):
    """
    Выполняет предсказание на основе обученной модели

    model_name: имя модели для загрузки
    df: DataFrame с входными данными
    Возвращает список предсказаний в формате словаря
    """
    try:
        model = load_model(model_name)
        logger.info(f"Модель {model_name} успешно загружена")

        df_processed = preprocess_data(df)
        logger.info("Данные успешно предобработаны")

        validate_required_columns(df_processed, REQUIRED_COLUMNS)
        logger.info("Все необходимые столбцы присутствуют в данных")

        predictions = model.predict(df_processed[REQUIRED_COLUMNS])
        logger.info("Предсказания успешно выполнены")

        df_processed['prediction'] = predictions
        return df_processed.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказания для модели {model_name}: {e}")
        raise RuntimeError(f"Ошибка при выполнении предсказания: {e}")


def get_model_info(model_name: str):
    """
    Возвращает информацию о модели, включая метрики и кривую обучения

    model_name: имя модели для загрузки
    Возвращает словарь с метриками и кривой обучения в формате base64
    """
    try:
        metrics = load_metrics(model_name)
        logger.info(f"Метрики для модели {model_name} успешно загружены")

        learning_curve_path = os.path.join(MODEL_DIR, model_name, "learning_curve.png")

        if os.path.exists(learning_curve_path):
            with open(learning_curve_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            logger.info(f"Кривая обучения для модели {model_name} успешно загружена")
        else:
            img_base64 = None
            logger.warning(f"Кривая обучения для модели {model_name} не найдена")

        return {
            "metrics": metrics,
            "learning_curve": img_base64
        }
    except Exception as e:
        logger.error(f"Ошибка при получении информации о модели {model_name}: {e}")
        raise RuntimeError(f"Ошибка при получении информации о модели: {e}")
