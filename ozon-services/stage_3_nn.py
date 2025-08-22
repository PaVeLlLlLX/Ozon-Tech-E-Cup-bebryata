import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import optuna

def train_final_model():
    SEED = 42
    N_TRIALS = 30 # Количество попыток для подбора параметров. Можете уменьшить до 10-15 для скорости.
    
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

    print(f"\n--- Начало подбора гиперпараметров ({N_TRIALS} попыток) ---")

    def _objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "depth": trial.suggest_int("depth", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 64, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "scale_pos_weight": y_train.value_counts()[0] / y_train.value_counts()[1],
            "loss_function": "Logloss",
            "eval_metric": "F1",
            "random_seed": SEED,
            "task_type": "GPU",
            "verbose": 0,
        }
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            use_best_model=True
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=N_TRIALS)

    print("\nПодбор параметров завершен!")
    print(f"Лучший F1-score на валидации: {study.best_value:.4f}")
    print("Лучшие параметры:")
    print(study.best_params)

    print("\n--- Обучение финальной модели на лучших параметрах ---")
    
    # Добавляем к лучшим параметрам обязательные, которые не подбирались
    final_params = study.best_params
    final_params.update({
        "scale_pos_weight": y_train.value_counts()[0] / y_train.value_counts()[1],
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "random_seed": SEED,
        "task_type": "GPU",
        "verbose": 100,
        "scale_pos_weight": y_train.value_counts()[0] / y_train.value_counts()[1],
    })

    model = CatBoostClassifier(**final_params)
    
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

    # --- Формирование submission.csv ---

    print("\n--- Создание предсказаний для тестового набора ---")

    X_test = df_test_raw.join(test_neural)

    training_columns = model.get_feature_names()
    X_test = X_test[training_columns]
    
    print(f"Финальный размер тестовых данных: {X_test.shape}")

    print("Получение вероятностей для тестовых данных...")
    test_probabilities = model.predict_proba(X_test, ntree_start=0, ntree_end=best_iteration)[:, 1]

    print(f"Применение порога {THRESHOLD}...")
    test_predictions = (test_probabilities >= THRESHOLD).astype(int)

    submission_df = pd.DataFrame({
        'id': X_test.index,
        'resolution': test_predictions
    })
    
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nГотово! Файл с предсказаниями сохранен в {submission_path}")
    print("Распределение предсказанных классов в submission файле:")
    print(submission_df['resolution'].value_counts(normalize=True))
if __name__ == '__main__':
    train_final_model()