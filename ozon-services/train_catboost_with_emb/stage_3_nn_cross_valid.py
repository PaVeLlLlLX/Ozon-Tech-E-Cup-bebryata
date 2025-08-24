# stage_3_train_final_catboost.py

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, classification_report

def train_with_cross_validation():
    print("--- ЭТАП 3: Обучение с кросс-валидацией и создание предсказаний ---")
    
    SEED = 42
    N_FOLDS = 10 
    PREDICTION_THRESHOLD = 0.9 

    CATBOOST_PARAMS = {
        'iterations': 3000,
        'learning_rate': 0.05,
        'depth': 8,
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'random_seed': SEED,
        'task_type': "GPU",
        'verbose': 0,
        'early_stopping_rounds': 150
    }

    print("Загрузка данных...")
    
    df_folds = pd.read_csv("../data/train_with_folds.csv", index_col='id')
    df_test_raw = pd.read_csv("../data/processed/test.csv", index_col='id')

    train_neural = pd.read_parquet("train_neural_features.parquet")
    val_neural = pd.read_parquet("val_neural_features.parquet")
    test_neural = pd.read_parquet("test_neural_features.parquet")

    full_train_val_neural = pd.concat([train_neural, val_neural])
    
    features_to_drop = ['resolution', 'fold', 'name_rus', 'ItemID', 'description']
    tabular_cols = [col for col in df_folds.columns if col not in features_to_drop]
    
    full_train_features = df_folds[tabular_cols].join(full_train_val_neural)
    
    test_tabular = df_test_raw[tabular_cols]
    X_test_final = test_tabular.join(test_neural)

    oof_preds = np.zeros(len(full_train_features))
    test_preds_list = []

    for fold in range(N_FOLDS):
        print(f"\n===== Обучение на фолде {fold} =====")
        
        train_ids = df_folds[df_folds['fold'] != fold].index
        val_ids = df_folds[df_folds['fold'] == fold].index
        
        X_train, y_train = full_train_features.loc[train_ids], df_folds.loc[train_ids, 'resolution']
        X_val, y_val = full_train_features.loc[val_ids], df_folds.loc[val_ids, 'resolution']
        
        neg_count, pos_count = y_train.value_counts().sort_index()
        CATBOOST_PARAMS['scale_pos_weight'] = neg_count / pos_count
        
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        best_iter = model.get_best_iteration()
        
        oof_fold_preds = model.predict_proba(X_val, ntree_end=best_iter)[:, 1]
        oof_preds[df_folds.index.isin(val_ids)] = oof_fold_preds
        
        test_fold_preds = model.predict_proba(X_test_final, ntree_end=best_iter)[:, 1]
        test_preds_list.append(test_fold_preds)

        print(f"F1 на фолде {fold}: {f1_score(y_val, (oof_fold_preds >= 0.5).astype(int)):.4f}")

    print("\n===== Кросс-валидация завершена =====")

    final_oof_f1 = f1_score(df_folds['resolution'], (oof_preds >= 0.5).astype(int))
    print(f"Общий F1-score по всем OOF-предсказаниям (порог 0.5): {final_oof_f1:.4f}")

    final_oof_f1_thresholded = f1_score(df_folds['resolution'], (oof_preds >= PREDICTION_THRESHOLD).astype(int))
    print(f"Общий F1-score по всем OOF-предсказаниям (порог {PREDICTION_THRESHOLD}): {final_oof_f1_thresholded:.4f}")
    
    print(f"\nОтчет по классификации для OOF-предсказаний (порог {PREDICTION_THRESHOLD}):")
    print(classification_report(df_folds['resolution'], (oof_preds >= PREDICTION_THRESHOLD).astype(int)))

    print("\nСоздание файла с предсказаниями...")
    
    avg_test_preds = np.mean(test_preds_list, axis=0)
    
    final_test_predictions = (avg_test_preds >= PREDICTION_THRESHOLD).astype(int)
    
    submission_df = pd.DataFrame({
        'id': X_test_final.index,
        'prediction': final_test_predictions
    })
    
    submission_path = "submission_c_v.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nГотово! Файл с предсказаниями сохранен в {submission_path}")
    print("Распределение предсказанных классов в submission файле:")
    print(submission_df['resolution'].value_counts(normalize=True))


if __name__ == '__main__':
    train_with_cross_validation()