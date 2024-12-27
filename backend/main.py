import base64
import io
import json
import os

import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from backend.config import MODEL_DIR
from backend.services.eda_service import get_plots
from backend.services.training_service import train_model
from backend.services.prediction_service import predict, get_model_info
from backend.managers.model_manager import list_available_models

import warnings
import logging

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings('ignore')
app = FastAPI()

LOCAL_FILE_PATH = "data/credit_card_transactions.csv"  # Локальный путь к файлам


class ModelParams(BaseModel):
    max_iter: int = 1000
    C: float = 1.0


@app.post("/train")
async def train_and_save_model(
    file: UploadFile = File(None),
    file_path: str = Form(None),
    use_local_file: bool = Form(False),
    max_iter: int = Form(1000),
    C: float = Form(1.0)
):
    """
    Обучение модели на предоставленных данных и сохранение результатов.
    """
    model_params = ModelParams(max_iter=max_iter, C=C)
    try:
        if use_local_file:
            if os.path.exists(LOCAL_FILE_PATH):
                df = pd.read_csv(LOCAL_FILE_PATH)
                logger.info(f"Данные загружены из локального файла: {LOCAL_FILE_PATH}")
            else:
                logger.error("Локальный файл не найден")
                raise HTTPException(status_code=404, detail="Локальный файл не найден")
        elif file:
            df = pd.read_csv(io.StringIO(str(file.file.read(), 'utf-8')))
            logger.info("Данные успешно загружены из загруженного файла")
        elif file_path:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                logger.info(f"Данные успешно загружены из пути: {file_path}")
            else:
                logger.error(f"Файл не найден: {file_path}")
                raise HTTPException(status_code=404, detail="Файл не найден")
        else:
            logger.error("Не передан файл для обучения")
            raise HTTPException(status_code=400, detail="Не передан файл")

        model_name = train_model(df, model_params)
        logger.info(f"Модель {model_name} успешно обучена и сохранена")
        model_info = get_model_info(model_name)

        return {"message": f"Модель {model_name} успешно обучена и сохранена.", "model_name": model_name, **model_info}
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели: {e}")


@app.post("/predict")
async def make_prediction(
    model_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Выполнение предсказаний на основе загруженной модели.
    """
    try:
        df = pd.read_csv(io.StringIO(str(file.file.read(), 'utf-8')))
        logger.info(f"Данные для предсказания успешно загружены для модели {model_name}")
        predictions = predict(model_name, df)
        logger.info(f"Предсказания успешно выполнены для модели {model_name}")
        return predictions
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказания для модели {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при выполнении предсказания: {e}")


@app.get("/model_info/{model_name}")
async def get_model_details(model_name: str):
    """
    Получение информации о модели, включая метрики и кривую обучения.
    """
    try:
        model_info = get_model_info(model_name)
        learning_curve_path = os.path.join(MODEL_DIR, model_name, "learning_curve.png")

        if os.path.exists(learning_curve_path):
            with open(learning_curve_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            model_info["learning_curve"] = img_base64
            logger.info(f"Кривая обучения успешно загружена для модели {model_name}")
        else:
            logger.warning(f"Кривая обучения не найдена для модели {model_name}")

        return model_info
    except Exception as e:
        logger.error(f"Ошибка при получении информации о модели {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении информации о модели: {e}")


@app.get("/models")
async def get_available_models():
    """
    Получение списка доступных моделей.
    """
    try:
        models = list_available_models()
        logger.info("Список доступных моделей успешно получен")
        return models
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении списка моделей: {e}")


@app.post("/eda")
async def perform_eda(file: UploadFile = File(...)):
    """
    Выполнение анализа данных (EDA) на основе загруженного файла.
    """
    try:
        plots = get_plots(file)
        logger.info("EDA успешно выполнен")
        return {"plots": plots}
    except Exception as e:
        logger.error(f"Ошибка при выполнении EDA: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Ошибка при выполнении EDA: {str(e)}"})
