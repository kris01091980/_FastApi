import logging
from datetime import datetime

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from preprocessing import preprocess_data
from managers.model_manager import save_model
from managers.metrics import save_metrics
from managers.visualizations import save_learning_curve
from config import REQUIRED_COLUMNS, TARGET_COLUMN

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_model(df, model_params):
    """
    Обучает модель логистической регрессии на предоставленных данных.

    df: DataFrame с исходными данными
    model_params: параметры модели, включая max_iter и C
    Возвращает имя сохранённой модели
    """
    try:
        # Преобразуем данные
        df_processed = preprocess_data(df)
        logger.info("Данные успешно предобработаны")

        X = df_processed[REQUIRED_COLUMNS]
        y = df_processed[TARGET_COLUMN]

        # Разделяем данные на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Данные успешно разделены на тренировочную и тестовую выборки")

        # Инициализируем модель
        model = LogisticRegression(max_iter=model_params.max_iter, C=model_params.C)
        logger.info(f"Инициализирована модель {model.__class__.__name__} с параметрами: {model_params}")

        # Обучаем модель
        model.fit(X_train, y_train)
        logger.info(f"Модель {model.__class__.__name__} успешно обучена")

        # Генерируем имя модели
        model_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Сохраняем модель
        save_model(model, model_name)
        logger.info(f"Модель сохранена с именем {model_name}")

        # Сохраняем метрики
        metrics = {
            "accuracy_train": accuracy_score(y_train, model.predict(X_train)),
            "accuracy_test": accuracy_score(y_test, model.predict(X_test)),
            "f1_train": f1_score(y_train, model.predict(X_train), average='weighted'),
            "f1_test": f1_score(y_test, model.predict(X_test), average='weighted'),
            "precision_train": precision_score(y_train, model.predict(X_train), average='weighted'),
            "precision_test": precision_score(y_test, model.predict(X_test), average='weighted'),
            "recall_train": recall_score(y_train, model.predict(X_train), average='weighted'),
            "recall_test": recall_score(y_test, model.predict(X_test), average='weighted'),
        }

        save_metrics(metrics, model_name)
        logger.info(f"Метрики сохранены для модели {model_name}")

        # Сохраняем кривую обучения
        save_learning_curve(X_train, y_train, model, model_name)
        logger.info(f"Кривая обучения сохранена для модели {model_name}")

        return model_name
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        raise RuntimeError(f"Ошибка при обучении модели: {e}")
