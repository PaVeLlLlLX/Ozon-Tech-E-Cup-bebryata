import pandas as pd
from sklearn.model_selection import StratifiedKFold


INPUT_CSV_PATH = "../data/processed/train.csv" 
OUTPUT_CSV_PATH = "../data/train_with_folds.csv"
N_SPLITS = 10
SEED = 42

df = pd.read_csv(INPUT_CSV_PATH)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

df["fold"] = -1

for fold_num, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df['resolution'])):
    df.loc[val_idx, "fold"] = fold_num

df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Файл с фолдами сохранен в {OUTPUT_CSV_PATH}")
print("Распределение по фолдам:")
print(df['fold'].value_counts())