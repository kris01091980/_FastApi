import matplotlib.pyplot as plt
import matplotlib
import os
import logging

import numpy as np
from sklearn.model_selection import learning_curve

from config import MODEL_DIR

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

matplotlib.use('TkAgg')


def save_learning_curve(X, y, model, model_name: str):
    """
    Создает и сохраняет кривую обучения для заданной модели.

    X: признаки для обучения
    y: целевая переменная
    model: модель, для которой строится кривая обучения
    model_name: имя модели, используемое для сохранения
    """
    try:
        # Получаем данные для кривой обучения
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        # Вычисляем средние значения для каждой из кривых
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        # Строим график
        plt.figure()
        plt.plot(train_sizes, train_scores_mean, label="Train score", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Test score", color="g")
        plt.xlabel("Training Set Size")
        plt.ylabel("Score")
        plt.title(f"Learning Curve for {model_name}")
        plt.legend()

        # Сохраняем график
        learning_curve_path = os.path.join(MODEL_DIR, model_name, "learning_curve.png")
        os.makedirs(os.path.dirname(learning_curve_path), exist_ok=True)
        plt.savefig(learning_curve_path)
        plt.close()

        logger.info(f"Кривая обучения для модели {model_name} успешно сохранена в {learning_curve_path}")
    except Exception as e:
        logger.error(f"Ошибка при создании кривой обучения для модели {model_name}: {e}")
        raise RuntimeError(f"Ошибка при создании кривой обучения: {e}")
