import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

def train_final_model():
    SEED = 42
    
    df_train_val_raw = pd.read_csv("../data/train_with_folds.csv", index_col='id')
    # df_test_raw = pd.read_csv("path/to/your/test.csv", index_col='id') # для предсказаний

    train_neural = pd.read_parquet("train_neural_features.parquet")
    val_neural = pd.read_parquet("val_neural_features.parquet")

    VAL_FOLD = 0
    train_ids = df_train_val_raw[df_train_val_raw['fold'] != VAL_FOLD].index
    val_ids = df_train_val_raw[df_train_val_raw['fold'] == VAL_FOLD].index

    features_to_drop = ['resolution', 'name_rus', 'ItemID', 'description']
    tabular_cols = [col for col in df_train_val_raw.columns if col not in features_to_drop]

    X_train_tab = df_train_val_raw.loc[train_ids, tabular_cols]
    X_train_neural = train_neural.loc[train_ids]
    X_train = X_train_tab.join(X_train_neural)
    y_train = df_train_val_raw.loc[train_ids, 'resolution']

    # Validation
    X_val_tab = df_train_val_raw.loc[val_ids, tabular_cols]
    X_val_neural = val_neural.loc[val_ids]
    X_val = X_val_tab.join(X_val_neural)
    y_val = df_train_val_raw.loc[val_ids, 'resolution']

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        
        loss_function='Logloss',
        eval_metric='F1',
        scale_pos_weight=neg_count / pos_count,
        random_seed=SEED,
        task_type="GPU",
        verbose=100
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=150
    )
    preds = model.predict(X_val)
    print("\nОтчет по классификации на валидационной выборке:")
    print(classification_report(y_val, preds))
    
    model.save_model("final_catboost_model.cbm")
    print("\nФинальная модель сохранена.")

    THRESHOLD = 0.9
    print("\nОтчет по классификации на валидационной выборке с порогом:")
    val_probabilities = model.predict_proba(X_val)[:, 1]

    preds_with_threshold = (val_probabilities >= THRESHOLD).astype(int)

    print(classification_report(y_val, preds_with_threshold))
if __name__ == '__main__':
    train_final_model()