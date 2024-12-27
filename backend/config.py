import os

MODEL_DIR = os.getenv("MODEL_DIR", "models")
REQUIRED_COLUMNS = [
    'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'trans_minute', 'trans_second',
    'trans_weekday', 'category', 'amt', 'gender', 'state', 'lat', 'long', 'city_pop',
    'merch_lat', 'merch_long', 'is_weekend', 'is_night', 'card_holder_age'
]
TARGET_COLUMN = 'is_fraud'
