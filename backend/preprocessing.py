from sklearn.preprocessing import LabelEncoder
import pandas as pd


def preprocess_data(df):
    """
    Функция для предобработки данных
    Принимает сырые данные и возвращает предобработанные данные
    """
    try:
        # Преобразование даты и времени
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['trans_year'] = df['trans_date_trans_time'].dt.year
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df['trans_day'] = df['trans_date_trans_time'].dt.day
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_minute'] = df['trans_date_trans_time'].dt.minute
        df['trans_second'] = df['trans_date_trans_time'].dt.second
        df['trans_weekday'] = df['trans_date_trans_time'].dt.weekday
        df = df.drop(columns=['trans_date_trans_time'])
    except Exception as e:
        raise ValueError(f"Ошибка при преобразовании даты и времени: {e}")

    try:
        # Удаление ненужных столбцов
        columns_to_drop = [
            'Unnamed: 0', 'first', 'last', 'street', 'city', 'zip',
            'trans_num', 'merch_zipcode', 'cc_num', 'merchant', 'job'
        ]
        df = df.drop(columns=columns_to_drop, errors='ignore')
    except Exception as e:
        raise ValueError(f"Ошибка при удалении столбцов: {e}")

    try:
        # Кодирование категориальных переменных
        cat_features = ['category', 'gender', 'state']
        for col in cat_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    except Exception as e:
        raise ValueError(f"Ошибка при кодировании категориальных переменных: {e}")

    try:
        # Добавление признака "выходной день"
        df['is_weekend'] = df['trans_weekday'].apply(lambda x: 1 if x >= 5 else 0)
    except Exception as e:
        raise ValueError(f"Ошибка при добавлении признака 'выходной день': {e}")

    try:
        # Добавление признака "ночное время"
        df['is_night'] = df['trans_hour'].apply(lambda x: 1 if x < 6 or x >= 22 else 0)
    except Exception as e:
        raise ValueError(f"Ошибка при добавлении признака 'ночное время': {e}")

    try:
        # Расчет возраста клиента
        df['dob'] = pd.to_datetime(df['dob'])
        df['birth_year'] = df['dob'].dt.year
        df['card_holder_age'] = df['trans_year'] - df['birth_year']
        df = df.drop(columns=['dob', 'birth_year'])
    except Exception as e:
        raise ValueError(f"Ошибка при расчете возраста клиента: {e}")

    try:
        # Удаление выбросов (опционально)
        outlier_threshold = 2700
        df = df[df['amt'] <= outlier_threshold]
    except Exception as e:
        raise ValueError(f"Ошибка при удалении выбросов: {e}")

    return df
