2. Использование сервиса**

##### **API через FastAPI (backend)**:
1. **Обучение модели:**
   - Отправьте POST-запрос на `/train` с CSV-файлом данных.
   - Пример использования через `curl`:
     ```bash
     curl -X POST "http://localhost:8000/train" \
          -F "file=@path/to/your_file.csv" \
          -F "max_iter=1000" \
          -F "C=1.0"
     ```
   - Ответ вернёт имя модели, метрики и кривую обучения.

2. **Предсказания:**
   - Отправьте POST-запрос на `/predict` с моделью и данными.
   - Пример:
     ```bash
     curl -X POST "http://localhost:8000/predict" \
          -F "model_name=LogisticRegression_YYYYMMDD_HHMMSS" \
          -F "file=@path/to/your_file.csv"
     ```

3. **Просмотр информации о модели:**
   - Отправьте GET-запрос на `/model_info/{model_name}`:
     ```bash
     curl -X GET "http://localhost:8000/model_info/LogisticRegression_YYYYMMDD_HHMMSS"
     ```

4. **Просмотр списка доступных моделей:**
   - Отправьте GET-запрос на `/models`:
     ```bash
     curl -X GET "http://localhost:8000/models"
     ```

5. **EDA (Exploratory Data Analysis):**
   - Отправьте POST-запрос на `/eda` с файлом:
     ```bash
     curl -X POST "http://localhost:8000/eda" \
          -F "file=@path/to/your_file.csv"
     ```

---
