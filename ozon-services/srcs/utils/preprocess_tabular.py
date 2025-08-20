# ozon-services/srcs/utils/preprocess_tabular.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from Levenshtein import distance as lev_distance
from preprocess_text import clean_description, clean_name

# --- Класс для Target Encoding ---
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=20.0):
        self.smoothing = smoothing
        self.mappings_ = {}

    def fit(self, X, y):
        X_temp = X.copy()
        for col in X_temp.columns:
            global_mean = y.mean()
            full_mapping = y.groupby(X_temp[col]).mean()
            n = y.groupby(X_temp[col]).count()
            
            smooth_mapping = (full_mapping * n + global_mean * self.smoothing) / (n + self.smoothing)
            self.mappings_[col] = smooth_mapping
            self.mappings_[f'{col}_global_mean'] = global_mean
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame(index=X.index)
        for col in X.columns:
            mapping = self.mappings_.get(col)
            global_mean = self.mappings_.get(f'{col}_global_mean')
            if mapping is not None:
                X_transformed[f'{col}_te'] = X[col].map(mapping).fillna(global_mean)
            else:
                # Если для колонки нет маппинга (редкий случай), заполняем средним
                X_transformed[f'{col}_te'] = global_mean
        return X_transformed

# --- Функции для создания признаков ---
def brand_match_score(row):
    # Работа с ОЧИЩЕННЫМИ данными
    name = str(row['name_rus_cleaned']) # Используем очищенное название
    brand = str(row['brand_name_cleaned']) # Используем очищенный бренд
    if not brand: return -1
    if brand in name: return 1
    try:
        min_dist = min([lev_distance(brand, word) for word in name.split()])
        if min_dist <= 2: return 0.5
    except (ValueError, TypeError): return 0
    return 0

def create_features(df):
    """Добавляет в DataFrame новые признаки, созданные на основе EDA."""
    print("Очистка текстовых описаний...")
    df['description_cleaned'] = df['description'].apply(clean_description)
    df['name_rus_cleaned'] = df['name_rus'].apply(clean_description)
    df['brand_name_cleaned'] = df['brand_name'].apply(clean_name)

    print("Создание новых признаков...")
    
    median_price_by_cat = df.groupby('CommercialTypeName4')['PriceDiscounted'].transform('median')
    df['price_vs_cat_median'] = df['PriceDiscounted'] / (median_price_by_cat + 1)
    
    df['name_len'] = df['name_rus'].str.len().fillna(0)
    df['name_word_count'] = df['name_rus'].str.split().str.len().fillna(0)
    
    KEYWORDS = ['копия', 'реплика', 'аналог', 'совместимый']
    for word in KEYWORDS:
        df[f'has_{word}'] = df['description'].str.contains(word, case=False).fillna(False).astype(int)
        
    df['brand_in_name_score'] = df.apply(brand_match_score, axis=1)
    df['has_ratings'] = df['rating_1_count'].notna().astype(int)
    df['has_brand'] = df['brand_name'].notna().astype(int)
    df['has_fake_returns'] = (df['item_count_fake_returns90'] > 0).astype(int)
    return df

def process_data(is_train=True):
    """Основная функция для обработки данных и сохранения артефактов."""
    
    base_data_path = 'data/'
    if is_train:
        print("Обработка TRAIN данных...")
        df = pd.read_csv(os.path.join(base_data_path, 'ml_ozon_сounterfeit_train.csv'))
    else:
        print("Обработка TEST данных...")
        df = pd.read_csv(os.path.join(base_data_path, 'ml_ozon_сounterfeit_test.csv'))


    # Сохраняем ключевые колонки, которые нужно оставить "как есть"
    key_cols = ['id', 'name_rus', 'ItemID']
    if is_train:
        key_cols.append('resolution')
    df_keys = df[key_cols].copy()

    print("Явная обработка пропусков...")
    
    # Категориальные
    df['brand_name'] = df['brand_name'].fillna('__MISSING__')
    df['CommercialTypeName4'] = df['CommercialTypeName4'].fillna('__MISSING__')
    df['description'] = df['description'].fillna('__MISSING__')

    df_keys['description'] = df['description']

    # Числовые, где NaN означает 0 (рейтинги, комментарии)
    rating_cols = [col for col in df.columns if 'rating' in col or 'count' in col or 'Count' in col]
    df[rating_cols] = df[rating_cols].fillna(0)

    # Числовые, где NaN означает "активности не было" (продажи, возвраты)
    activity_cols = [col for col in df.columns if 'Gmv' in col or 'Exemplar' in col or 'Order' in col]
    df[activity_cols] = df[activity_cols].fillna(0)

    df = create_features(df)
    
    target_encode_cols = ['brand_name', 'SellerID', 'CommercialTypeName4']
    
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    
    cols_to_remove_from_numeric = ['id', 'resolution', 'ItemID', 'SellerID']
    numeric_cols_to_scale = [col for col in numeric_cols if col not in cols_to_remove_from_numeric]

    df['SellerID'] = df['SellerID'].astype(str)
    
    if is_train:
        print("Обучение препроцессора...")
        cols_to_drop = [
            'resolution', 'description', 'name_rus', 'ItemID',
            'description_cleaned', 'name_rus_cleaned', 'brand_name_cleaned'
        ]
        X = df.drop(columns=cols_to_drop, errors='ignore')
        y = df['resolution']
        
        # Обучаем Scaler
        scaler = StandardScaler()
        scaler.fit(X[numeric_cols_to_scale])
        
        # Обучаем Target Encoder
        target_encoder = TargetEncoder()
        target_encoder.fit(X[target_encode_cols], y)
        
        # Сохраняем оба в виде словаря
        preprocessor = {'scaler': scaler, 'target_encoder': target_encoder}
        print("Сохранение артефакта препроцессора в 'artifacts/preprocessor.pkl'...")
        os.makedirs('artifacts', exist_ok=True)
        with open("artifacts/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
    else:
        print("Загрузка обученного препроцессора...")
        with open("artifacts/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        cols_to_drop = [
            'description', 'name_rus', 'ItemID',
            'description_cleaned', 'name_rus_cleaned', 'brand_name_cleaned'
        ]
        X = df.drop(columns=cols_to_drop, errors='ignore')

    print("Трансформация данных и добавление новых признаков...")
    
    # 1. Применяем Scaler и добавляем отмасштабированные признаки с новым суффиксом
    scaled_data = scaler.transform(X[numeric_cols_to_scale])
    df_scaled = pd.DataFrame(scaled_data, columns=[f"{c}_scaled" for c in numeric_cols_to_scale], index=X.index)
    
    # 2. Применяем Target Encoder, он вернет DataFrame с _te суффиксами
    encoded_data = target_encoder.transform(X[target_encode_cols])
    
    # 3. Объединяем все в один большой DataFrame: исходный + отмасштабированные + закодированные
    df_final = pd.concat([df_keys, df_scaled, encoded_data], axis=1)
    
    if is_train:
        output_path = 'data/processed/train.csv'
    else:
        output_path = 'data/processed/test.csv'
        
    os.makedirs('data/processed', exist_ok=True)
    # Сохраняем в CSV
    df_final.to_csv(output_path, index=False)
    
    print(f"Обработка завершена! Результат сохранен в {output_path}")
    print(f"Итоговая форма DataFrame: {df_final.shape}")
    print("Пример колонок:", list(df_final.columns[:5]), "...", list(df_final.columns[-5:]))



if __name__ == '__main__':
    # Устанавливаем правильную рабочую директорию, если запускаем из ozon-services/srcs/utils
    # Это сделано для того, чтобы пути 'data/' и 'artifacts/' работали корректно
    if os.getcwd().endswith('utils'):
        os.chdir('../../')
        print(f"Рабочая директория изменена на: {os.getcwd()}")

    process_data(is_train=True)
    process_data(is_train=False)