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
    name = str(row['name_rus']).lower()
    brand = str(row['brand_name']).lower()
    if brand == 'nan' or brand == '__missing__': return -1
    if brand in name: return 1
    try:
        min_dist = min([lev_distance(brand, word) for word in name.split()])
        if min_dist <= 2: return 0.5
    except (ValueError, TypeError): return 0
    return 0

def create_features(df):
    """Добавляет в DataFrame новые признаки, созданные на основе EDA."""
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
        df = pd.read_csv(os.path.join(base_data_path, 'ml_ozon_сounterfeit_train.csv'), index_col='id')
    else:
        print("Обработка TEST данных...")
        df = pd.read_csv(os.path.join(base_data_path, 'ml_ozon_сounterfeit_test.csv'), index_col='id')

    # Заполняем пропуски в brand_name ДО создания признаков
    df['brand_name'] = df['brand_name'].fillna('__MISSING__')
    df = create_features(df)
    
    target_encode_cols = ['brand_name', 'SellerID', 'CommercialTypeName4']
    
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    
    cols_to_remove_from_numeric = ['resolution', 'ItemID', 'SellerID']
    numeric_cols = [col for col in numeric_cols if col not in cols_to_remove_from_numeric]

    df['SellerID'] = df['SellerID'].astype(str)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    target_transformer = Pipeline(steps=[
        ('target_encoder', TargetEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('target', target_transformer, target_encode_cols)
        ],
        remainder='drop'
    )

    if is_train:
        print("Обучение препроцессора...")
        X = df.drop(columns=['resolution', 'description', 'name_rus', 'ItemID'], errors='ignore')
        y = df['resolution']
        preprocessor.fit(X, y)
        
        print("Сохранение артефакта препроцессора в 'artifacts/preprocessor.pkl'...")
        os.makedirs('artifacts', exist_ok=True)
        with open("artifacts/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
    else:
        print("Загрузка обученного препроцессора...")
        with open("artifacts/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        X = df.drop(columns=['description', 'name_rus', 'ItemID'], errors='ignore')

    print("Трансформация данных...")
    X_transformed = preprocessor.transform(X)
    
    new_cols = numeric_cols + [f"{c}_te" for c in target_encode_cols]
    df_final = pd.DataFrame(X_transformed, columns=new_cols, index=df.index)
    
    if is_train:
        df_final['resolution'] = df['resolution']
        output_path = 'data/processed/train.parquet'
    else:
        output_path = 'data/processed/test.parquet'
        
    os.makedirs('data/processed', exist_ok=True)
    df_final.to_parquet(output_path)
    
    print(f"Обработка завершена! Результат сохранен в {output_path}")
    print(f"Итоговая форма признаков: {df_final.shape}")
    print(df_final.head())


if __name__ == '__main__':
    # Устанавливаем правильную рабочую директорию, если запускаем из ozon-services/srcs/utils
    # Это сделано для того, чтобы пути 'data/' и 'artifacts/' работали корректно
    if os.getcwd().endswith('utils'):
        os.chdir('../../')
        print(f"Рабочая директория изменена на: {os.getcwd()}")

    process_data(is_train=True)
    process_data(is_train=False)