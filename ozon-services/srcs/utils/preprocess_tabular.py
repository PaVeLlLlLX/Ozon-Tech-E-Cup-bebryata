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
from sklearn.model_selection import StratifiedKFold
from preprocess_text import clean_description, clean_name_rus, clean_brand_commertial4_name
from util import delete_rows_without_images
from scipy.stats import entropy

# --- Новая функция для оптимизации памяти ---
def optimize_memory_usage(df):
    """
    Итерируется по всем колонкам DataFrame и изменяет тип данных
    для уменьшения использования памяти.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Использование памяти до оптимизации: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Используем float32 вместо float16 для большей стабильности вычислений
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Использование памяти после оптимизации: {end_mem:.2f} MB')
    print(f'Сокращение: {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

# --- Класс для Target Encoding ---
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=20.0, n_splits=5, random_state=42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean_ = 0
        self.mappings_ = {}
        self.encoded_cols_ = []

    def fit(self, X, y):
        self.global_mean_ = np.mean(y)
        self.encoded_cols_ = [f"{c}_te" for c in X.columns]
        
        # 1. Обучаем "боевые" маппинги на ВСЕХ данных для будущего transform (на тесте)
        for col in X.columns:
            full_mapping = y.groupby(X[col]).mean()
            n = y.groupby(X[col]).count()
            self.mappings_[col] = (full_mapping * n + self.global_mean_ * self.smoothing) / (n + self.smoothing)
        
        # 2. Создаем OOF (Out-of-Fold) признаки для ТРЕНИРОВОЧНОГО набора
        oof_encodings = pd.DataFrame(index=X.index)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in X.columns:
            encoded_col = pd.Series(index=X.index, dtype=float)
            for train_idx, val_idx in skf.split(X, y):
                # Считаем среднее только на обучающем фолде
                fold_mapping = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()
                n_fold = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).count()
                smooth_fold_mapping = (fold_mapping * n_fold + self.global_mean_ * self.smoothing) / (n_fold + self.smoothing)
                encoded_col.iloc[val_idx] = X[col].iloc[val_idx].map(smooth_fold_mapping)
            
            oof_encodings[f'{col}_te'] = encoded_col.fillna(self.global_mean_)
        
        self.oof_encodings_ = oof_encodings
        return self

    def transform(self, X):
        # Если мы на этапе fit, то возвращаем уже посчитанные OOF-признаки
        if hasattr(self, 'oof_encodings_') and X.index.equals(self.oof_encodings_.index):
            return self.oof_encodings_
        # Если на этапе predict (тестовые данные), применяем "боевые" маппинги
        else:
            X_encoded = pd.DataFrame(index=X.index)
            for col in X.columns:
                X_encoded[f'{col}_te'] = X[col].map(self.mappings_[col]).fillna(self.global_mean_)
            return X_encoded

    def get_feature_names_out(self, input_features=None):
        return self.encoded_cols_

# --- НОВАЯ, ОБЪЕДИНЕННАЯ ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ПРИЗНАКОВ ---
def generate_all_features(df):
    """
    Добавляет в DataFrame новые признаки, созданные на основе EDA и анализа важности.
    Объединяет старые и новые генераторы признаков.
    """
    print("Создание новых признаков...")
    

    '''

        --- Блок 1: Признаки на основе текста и базовые ---

    '''

    df['name_len'] = df['name_rus'].str.len().fillna(0)
    df['description_len'] = df['description'].str.len().fillna(0)
    df['name_word_count'] = df['name_rus'].str.split().str.len().fillna(0)
    df['description_word_count'] = df['description'].str.split().str.len().fillna(0)
    
    # Считаем долю заглавных букв. Используем исходный 'raw_name_rus', т.к. 'name_rus' уже в нижнем регистре.
    df['caps_ratio_in_name'] = df['raw_name_rus'].str.count(r'[A-ZА-Я]') / (df['raw_name_rus'].str.len() + 1e-6)

    # Количество цифр в названии (часто в подделках)
    df['digits_in_name_ratio'] = df['name_rus'].str.count(r'\d') / (df['name_len'] + 1e-6)
    
    KEYWORDS = ['копия', 'реплика', 'аналог', 'совместимый']
    for word in KEYWORDS:
        df[f'has_{word}_description'] = df['description'].str.contains(word, case=False).fillna(False).astype(int)
        df[f'has_{word}_name_rus'] = df['name_rus'].str.contains(word, case=False).fillna(False).astype(int)

    # Подозрительные ключевые слова для контрафакта
    fake_keywords = ['оригинал', 'гарантия', '100%', 'качество', 'новый', 'лицензия']
    df['fake_keywords_count'] = sum(
        df['description'].str.count(word, case=False) for word in fake_keywords
    )
    
    # Функция для brand_in_name_score
    def brand_match_score(row):
        name = str(row['name_rus'])
        brand = str(row['brand_name'])
        if not brand or brand == '__MISSING__': return -1
        if brand in name: return 1
        try:
            min_dist = min([lev_distance(brand, word) for word in name.split()])
            if min_dist <= 2: return 0.5
        except (ValueError, TypeError): return 0
        return 0
    df['brand_in_name_score'] = df.apply(brand_match_score, axis=1)

    # Jaccard similarity between name_rus and description words (high if redundant)
    def jaccard_sim(row):
        name_words = set(str(row['name_rus']).split())
        desc_words = set(str(row['description']).split())
        if not name_words or not desc_words: return 0
        return len(name_words & desc_words) / len(name_words | desc_words)
    df['name_desc_similarity'] = df.apply(jaccard_sim, axis=1)

    # Повторы слов (спам-паттерн)
    def count_word_repeats(text):
        if pd.isna(text): return 0
        words = str(text).lower().split()
        if len(words) <= 1: return 0
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        return sum(1 for count in word_counts.values() if count > 1) / len(words)
    df['word_repeat_ratio'] = df['description'].apply(count_word_repeats)

    # Special char ratio in name+desc (spam/fakes have more punctuation)
    special_chars = r'[!@#$%^&*()]'
    df['special_char_ratio'] = (
        (df['name_rus'].str.count(special_chars) + df['description'].str.count(special_chars)) /
        (df['name_len'] + df['description_len'] + 1e-6)
    )
    
    df['has_ratings'] = (df[['rating_1_count','rating_2_count','rating_3_count','rating_4_count','rating_5_count']].sum(axis=1) > 0).astype(int)
    df['has_brand'] = (df['brand_name'].notna() & (df['brand_name'] != '__MISSING__')).astype(int)
    df['has_fake_returns'] = (df['item_count_fake_returns90'] > 0).astype(int)
    df['has_photo'] = (df['photos_published_count'] > 0).astype(int)
    df['has_video'] = (df['videos_published_count'] > 0).astype(int)
    df['has_comments'] = (df['comments_published_count'] > 0).astype(int)

    premium_brands = ['apple', 'samsung', 'sony', 'dyson', 'logitech', 'jabra', 'nintendo', 'playstation', 'xbox'] # Можно еще какие-нибудь добавить
    df['is_premium_brand'] = df['brand_name'].apply(lambda x: 1 if str(x).lower() in premium_brands else 0)
    # Частота встречаемости бренда (редкие бренды более подозрительны)
    brand_freq = df['brand_name'].value_counts()
    df['brand_frequency'] = df['brand_name'].map(brand_freq)
    # Наличие технических характеристик в описании
    tech_terms = ['характеристики', 'технические', 'параметры', 'specifications', 'features']
    df['has_tech_specs'] = df['description'].apply(
        lambda x: 1 if any(term in str(x).lower() for term in tech_terms) else 0
    )
    # Длина описания относительно категории
    df['desc_len_vs_category'] = df.groupby('CommercialTypeName4')['description_len'].transform(
        lambda x: x / x.mean()
    ).fillna(0)

    '''

        --- Блок 2: Продвинутые ценовые признаки ---

    '''

    # Группируем по категориям и брендам, чтобы найти аномально дешевые/дорогие товары
    grp_cat = df.groupby('CommercialTypeName4')['PriceDiscounted']
    median_cat = grp_cat.transform('median')
    mad_cat = (df['PriceDiscounted'] - median_cat).abs().groupby(df['CommercialTypeName4']).transform('median') + 1e-6
    df['log_price'] = np.log1p(df['PriceDiscounted'])
    df['log_price_minus_cat_median'] = np.log1p(df['PriceDiscounted']) - np.log1p(median_cat + 1e-6)
    df['robust_price_z_cat'] = (df['PriceDiscounted'] - median_cat) / mad_cat
    # Price z-score per cat (std-based, if not robust)
    cat_mean = grp_cat.transform('mean')
    cat_std = grp_cat.transform('std') + 1e-6
    df['price_zscore_cat'] = (df['PriceDiscounted'] - cat_mean) / cat_std

    grp_brand = df.groupby('brand_name')['PriceDiscounted']
    median_brand = grp_brand.transform('median')
    mad_brand = (df['PriceDiscounted'] - median_brand).abs().groupby(df['brand_name']).transform('median') + 1e-6
    df['log_price_minus_brand_median'] = np.log1p(df['PriceDiscounted']) - np.log1p(median_brand + 1e-6)
    df['robust_price_z_brand'] = (df['PriceDiscounted'] - median_brand) / mad_brand
    # Price z-score per brand (analog to cat)
    brand_mean = grp_brand.transform('mean')
    brand_std = grp_brand.transform('std') + 1e-6
    df['price_zscore_brand'] = (df['PriceDiscounted'] - brand_mean) / brand_std

    # Price vs global median
    global_median = df['PriceDiscounted'].median() + 1
    df['price_vs_global_median'] = df['PriceDiscounted'] / global_median
    

    #df['price_vs_cat_median'] = df['PriceDiscounted'] / (df.groupby('CommercialTypeName4')['PriceDiscounted'].transform('median') + 1)
    #df['price_vs_brand_median'] = df['PriceDiscounted'] / (df.groupby('brand_name')['PriceDiscounted'].transform('median') + 1)
    
    '''

        --- Блок 3: Признаки-соотношения (Ratios) на уровне товара. Превращаем "мусорные" признаки в золото! ---

    '''
    safe_div = lambda a, b: a / (b + 1e-6)
    df['item_return_ratio_7'] = safe_div(df['item_count_returns7'], df['item_count_sales7'])
    df['item_return_ratio_30'] = safe_div(df['item_count_returns30'], df['item_count_sales30'])
    df['item_return_ratio_90'] = safe_div(df['item_count_returns90'], df['item_count_sales90'])
    df['item_fake_return_ratio_7'] = safe_div(df['item_count_fake_returns7'], df['item_count_returns7'])
    df['item_fake_return_ratio_30'] = safe_div(df['item_count_fake_returns30'], df['item_count_returns30'])
    df['item_fake_return_ratio_90'] = safe_div(df['item_count_fake_returns90'], df['item_count_returns90'])
    # Sales growth (sudden surge suspicious)
    df['sales_growth_7_30'] = safe_div(df['item_count_sales7'], df['item_count_sales30'])
    # Return recent ratio (recent spike)
    df['return_recent_ratio_7_90'] = safe_div(df['item_count_returns7'], df['item_count_returns90'])
    # Item age ratio (new item from new seller riskier)
    df['item_age_ratio'] = safe_div(df['item_time_alive'], df['seller_time_alive'])
    # Return value ratio (monetary loss rate)
    df['return_value_ratio_90'] = safe_div(df['ExemplarReturnedValueTotal90'], df['GmvTotal90'])
    # Item GMV share (if main revenue, less fake)
    df['item_gmv_share_90'] = safe_div(
        (df['item_count_sales90'] * df['PriceDiscounted']),
        df['GmvTotal90']
    )
    
    '''

        --- Блок 4: Паттерны в рейтингах ---

    '''

    rating_cols = ['rating_1_count', 'rating_2_count', 'rating_3_count', 'rating_4_count', 'rating_5_count']
    df['total_ratings'] = df[rating_cols].sum(axis=1)
    df['avg_rating'] = safe_div(
        (df['rating_1_count'] * 1 + df['rating_2_count'] * 2 + df['rating_3_count'] * 3 + df['rating_4_count'] * 4 + df['rating_5_count'] * 5),
        df['total_ratings']
    )
    # Interactions (simple, for boost)
    df['price_avg_rating_interact'] = df['PriceDiscounted'] * df['avg_rating']
    # Поляризация оценок: много 1 и 5, мало средних. Характерно для контрафакта.
    df['rating_polarization'] = safe_div(
        (df['rating_1_count'] + df['rating_5_count']),
        df['total_ratings']
    )
    # Negative ratio (focus on bad ratings)
    df['negative_ratio'] = safe_div(
        (df['rating_1_count'] + df['rating_2_count']),
        df['total_ratings']
    )

    # Rating variance (weighted var of ratings 1-5)
    def weighted_var(counts):
        ratings = np.array([1, 2, 3, 4, 5])
        total = counts.sum()
        if total == 0: return 0
        mean = (ratings * counts).sum() / total
        var = ((ratings - mean)**2 * counts).sum() / total
        return var
    df['rating_variance'] = df[rating_cols].apply(weighted_var, axis=1)
    
    '''

        --- Блок 5: Агрегированные признаки на уровне продавца (Seller-level) ---

    '''

    # Создаем "профиль риска" для каждого продавца
    seller_agg = df.groupby('SellerID').agg(
        seller_avg_item_price=('PriceDiscounted', 'mean'),
        seller_total_sales_90=('item_count_sales90', 'sum'),
        seller_total_returns_90=('item_count_returns90', 'sum'),
        seller_total_fake_returns_90=('item_count_fake_returns90', 'sum'),
        seller_avg_rating=('avg_rating', 'mean')
    ).reset_index()

    # Считаем коэффициенты возвратов для продавца в целом
    seller_agg['seller_return_rate_90'] = safe_div(seller_agg['seller_total_returns_90'], seller_agg['seller_total_sales_90'])
    seller_agg['seller_fake_return_rate_90'] = safe_div(seller_agg['seller_total_fake_returns_90'], seller_agg['seller_total_returns_90'])
    # Присоединяем агрегированные данные обратно к основному датафрейму
    df = pd.merge(df, seller_agg, on='SellerID', how='left')

    # Seller category entropy (diverse = legit)
    cat_counts = df.groupby('SellerID')['CommercialTypeName4'].value_counts(normalize=True)
    entropy_per_seller = cat_counts.groupby(level=0).apply(lambda x: entropy(x))
    df['seller_category_entropy'] = df['SellerID'].map(entropy_per_seller).fillna(0)
    # Seller item depth (stock depth)
    df['seller_item_depth'] = safe_div(df['ItemAvailableCount'], df['ItemVarietyCount'])
    # Seller newbie (new sellers riskier)
    df['seller_newbie'] = (df['seller_time_alive'] < 30).astype(int)
    # Brand fraud rate (mean resolution per brand, only train)
    if 'resolution' in df.columns: # !!! Возможна утечка !!!
        brand_fraud = df.groupby('brand_name')['resolution'].mean()
        df['brand_fraud_rate'] = df['brand_name'].map(brand_fraud).fillna(0)
    # Has multiple items (repeats per seller)
    df['has_multiple_items'] = (df.groupby('SellerID')['id'].transform('count') > 1).astype(int)

    # Асимметрия распределения оценок
    def rating_skewness(counts):
        ratings = np.array([1, 2, 3, 4, 5])
        total = counts.sum()
        if total == 0: return 0
        mean = (ratings * counts).sum() / total
        std = np.sqrt(((ratings - mean)**2 * counts).sum() / total)
        if std == 0: return 0
        skewness = ((ratings - mean)**3 * counts).sum() / (total * std**3)
        return skewness
    df['rating_skewness'] = df[rating_cols].apply(rating_skewness, axis=1)

    # Подозрительная концентрация продаж
    df['sales_spike_7d'] = safe_div(df['item_count_sales7'] * 7, df['item_count_sales30'])
    df['returns_spike_7d'] = safe_div(df['item_count_returns7'] * 7, df['item_count_returns30'])

    '''

        --- Блок 6: Категориальные взаимодействия --- 
    
    '''

    # Редкие комбинации бренд-категория (подозрительно)
    brand_cat_counts = df.groupby(['brand_name', 'CommercialTypeName4']).size()
    df['brand_cat_rarity'] = df.apply(
        lambda x: brand_cat_counts.get((x['brand_name'], x['CommercialTypeName4']), 0), axis=1
    )
    df['is_rare_brand_cat'] = (df['brand_cat_rarity'] <= 2).astype(int)
    # Несоответствие цены категории и бренду одновременно
    df['price_brand_cat_anomaly'] = (
        (df['robust_price_z_cat'] < -2) & (df['robust_price_z_brand'] < -2)
    ).astype(int)
    # Популярность бренда в категории
    brand_popularity = df.groupby(['CommercialTypeName4', 'brand_name']).size()
    cat_totals = df.groupby('CommercialTypeName4').size()
    df['brand_popularity_in_cat'] = df.apply(
        lambda x: brand_popularity.get((x['CommercialTypeName4'], x['brand_name']), 0) / 
              cat_totals.get(x['CommercialTypeName4'], 1), axis=1
    )

    # Заполняем пропуски в новых колонках нулями (на случай, если где-то деление на 0 дало NaN)
    new_feature_cols = [
        #'price_vs_cat_median', 'price_vs_brand_median', 
        'name_len', 'description_len', 'name_word_count', 'description_word_count', 'caps_ratio_in_name',
        'has_копия_description', 'has_копия_name_rus', 'has_реплика_description', 'has_реплика_name_rus',
        'has_аналог_description', 'has_аналог_name_rus', 'has_совместимый_description', 'has_совместимый_name_rus',
        'brand_in_name_score', 'name_desc_similarity', 'special_char_ratio',
        'has_ratings', 'has_brand', 'has_fake_returns', 'has_photo', 'has_video', 'has_comments',
        'log_price', 'log_price_minus_cat_median', 'robust_price_z_cat', 'log_price_minus_brand_median',
        'robust_price_z_brand', 'price_vs_global_median', 'price_zscore_cat', 'price_zscore_brand',
        'item_return_ratio_7', 'item_return_ratio_30', 'item_return_ratio_90',
        'item_fake_return_ratio_7', 'item_fake_return_ratio_30', 'item_fake_return_ratio_90',
        'sales_growth_7_30', 'return_recent_ratio_7_90', 'item_age_ratio', 'return_value_ratio_90', 'item_gmv_share_90',
        'total_ratings', 'avg_rating', 'price_avg_rating_interact', 'rating_polarization', 'negative_ratio', 'rating_variance',
        'seller_avg_item_price', 'seller_total_sales_90', 'seller_total_returns_90', 'seller_total_fake_returns_90',
        'seller_avg_rating', 'seller_return_rate_90', 'seller_fake_return_rate_90',
        'seller_category_entropy', 'seller_item_depth', 'seller_newbie', 'brand_fraud_rate', 'has_multiple_items', 
        'brand_frequency', 'is_premium_brand', 'rating_skewness', 'word_repeat_ratio', 'fake_keywords_count',
        'brand_cat_rarity', 'is_rare_brand_cat', 'price_brand_cat_anomaly', 'digits_in_name_ratio', 'has_tech_specs',
        'desc_len_vs_category', 'sales_spike_7d', 'returns_spike_7d', 'brand_popularity_in_cat'
    ]
    for col in new_feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"Создано {len(new_feature_cols)} новых признаков.")
    return df

# --- Новая функция для создания временных признаков ---
def create_ts_features(df):
    """
    Создает признаки, основанные на временной динамике поведения продавцов.
    ВАЖНО: предполагает, что df отсортирован по времени.
    """
    print("Создание временных (поведенческих) признаков...")
    
    # Сортируем на всякий случай, хотя трейн уже должен быть отсортирован
    df = df.sort_values(['SellerID','seller_time_alive','item_time_alive'], kind='mergesort')

    # кол-во ранее выставленных товаров
    df['seller_items_listed_expanding'] = df.groupby('SellerID').cumcount()

    # средняя цена предыдущих товаров продавца (строго прошлое)
    df['seller_avg_price_expanding'] = (
        df.groupby('SellerID')['PriceDiscounted']
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # отклонение текущей цены от исторического среднего продавца
    eps = 1e-6
    df['price_deviation_from_seller_mean'] = (
        (df['PriceDiscounted'] - df['seller_avg_price_expanding']) /
        (df['seller_avg_price_expanding'] + eps)
    ).fillna(0)

    # Динамика изменения рейтинга продавца
    df['seller_rating_trend'] = (
        df.groupby('SellerID')['avg_rating']
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Динамика изменения цен продавца
    df['seller_price_volatility'] = (
        df.groupby('SellerID')['PriceDiscounted']
        .apply(lambda s: s.shift(1).expanding().std())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # доля контрафакта у продавца в прошлом (только для train)
    if 'resolution' in df.columns:
        df['seller_fraud_rate_expanding'] = (
            df.groupby('SellerID')['resolution']
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
            .fillna(-1)
    )
    return df


def process_data(is_train=True):
    """Основная функция для обработки данных и сохранения артефактов."""
    
    base_data_path = './data/'
    if is_train:
        print("Обработка TRAIN данных...")
        base_data_path += 'train/'
        df = pd.read_csv(os.path.join(base_data_path, 'ml_ozon_сounterfeit_train.csv'))
    else:
        print("Обработка TEST данных...")
        base_data_path += 'test/'
        df = pd.read_csv(os.path.join(base_data_path, 'ml_ozon_сounterfeit_test.csv'))

    key_cols = ['id', 'name_rus', 'ItemID']
    if is_train:
        key_cols.append('resolution')

    print("Явная обработка пропусков...")
    num_cols_to_fill = [col for col in df.columns if 'rating' in col or 'count' in col or 'Count' in col or 'Gmv' in col or 'Exemplar' in col or 'Order' in col]
    df[num_cols_to_fill] = df[num_cols_to_fill].fillna(0)

    # [ИЗМЕНЕНО] Сохраняем исходное название товара ДО очистки для анализа регистра
    df['raw_name_rus'] = df['name_rus'].astype(str)

    print("Очистка текстовых описаний...")
    for col in ['description', 'name_rus', 'brand_name', 'CommercialTypeName4']:
        # Применяем соответствующую функцию очистки
        if col in ['brand_name', 'CommercialTypeName4']:
            df[col] = df[col].apply(clean_brand_commertial4_name)
        elif col == 'name_rus':
            df[col] = df[col].apply(clean_name_rus)
        else: # description
            df[col] = df[col].apply(clean_description)
        # Заполняем пустые строки после очистки
        df.loc[df[col] == '', col] = '__MISSING__'
    
    # [ИЗМЕНЕНО] Вызываем новую единую функцию для генерации всех признаков
    df = generate_all_features(df)
    
    # Вызываем функцию для создания временных признаков
    df = create_ts_features(df)

    # Удаляем временную колонку
    df = df.drop(columns=['raw_name_rus'])

    df_keys = df[key_cols].copy()
    df_keys['description'] = df['description']
    
    target_encode_cols = ['brand_name', 'SellerID', 'CommercialTypeName4']
    
    # Определяем числовые колонки для масштабирования. Теперь включаем все новые сгенерированные признаки.
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    cols_to_remove_from_numeric = ['id', 'resolution', 'ItemID', 'SellerID']
    numeric_cols_to_scale = [col for col in numeric_cols if col not in cols_to_remove_from_numeric]

    print(f"Количество признаков для масштабирования: {len(numeric_cols_to_scale)}")

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
            ('num', numeric_transformer, numeric_cols_to_scale),
            ('target', target_transformer, target_encode_cols)
        ],
        remainder='drop' # Явно указываем, что остальные колонки нужно отбросить
    )

    if is_train:
        print("Обучение препроцессора...")
        cols_to_drop_for_fit = ['resolution', 'description', 'name_rus', 'ItemID', 'id']
        X = df.drop(columns=cols_to_drop_for_fit, errors='ignore')
        y = df['resolution']
        preprocessor.fit(X, y)
        
        print("Сохранение артефакта препроцессора...")
        os.makedirs('artifacts', exist_ok=True)
        with open("artifacts/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
    else:
        print("Загрузка обученного препроцессора...")
        with open("artifacts/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

    print("Трансформация данных...")
    X_to_transform = df.drop(columns=['resolution', 'description', 'name_rus', 'ItemID', 'id'], errors='ignore')
    X_transformed = preprocessor.transform(X_to_transform)
    
    scaled_cols = [f"{c}_scaled" for c in numeric_cols_to_scale]
    encoded_cols = preprocessor.named_transformers_['target'].get_feature_names_out()
    
    # [ИЗМЕНЕНО] Собираем финальный список колонок. `remainder` теперь 'drop', поэтому `passthrough_cols` не нужны.
    new_cols = scaled_cols + list(encoded_cols)
    df_processed = pd.DataFrame(X_transformed, columns=new_cols, index=df.index)
    
    df_final = pd.merge(df_keys, df_processed, left_index=True, right_index=True)

    print("\nОптимизация использования памяти...")
    df_final = optimize_memory_usage(df_final)
    
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