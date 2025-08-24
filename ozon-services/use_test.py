# stage_4_semi_supervised_training.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from omegaconf import DictConfig
import hydra
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from srcs.utils import instantiate
import logging
from tqdm import tqdm

from srcs.model.model import MultimodalModel
from srcs.data_loader.data_loaders import OzonDataset, transform_train, transform_val
# ... другие импорты ...

@hydra.main(version_base=None, config_path='conf/', config_name='train')
def train_semi_supervised(config: DictConfig):
    print("--- ЭТАП 4: Полу-самостоятельное обучение ---")
    
    SEED = 42
    VAL_FOLD = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Загрузка и объединение признаков...")
    train_val_tabular = pd.read_csv("../data/train_with_folds.csv", index_col='id')
    test_tabular = pd.read_csv("../data/processed/test.csv", index_col='id')
    train_neural = pd.read_parquet("train_neural_features.parquet")
    val_neural = pd.read_parquet("val_neural_features.parquet")
    test_neural = pd.read_parquet("test_neural_features.parquet")
    df_folds = pd.read_csv("data/train_with_folds.csv", index_col='id')
    
    train_val_features = train_val_tabular.join(pd.concat([train_neural, val_neural]))
    X_test = test_tabular.join(test_neural)
    
    train_ids = df_folds[df_folds['fold'] != VAL_FOLD].index
    val_ids = df_folds[df_folds['fold'] == VAL_FOLD].index
    
    X_train = train_val_features.loc[train_ids]
    y_train = df_folds.loc[train_ids, 'resolution']
    X_val = train_val_features.loc[val_ids]
    y_val = df_folds.loc[val_ids, 'resolution']

    print("\nОбучение доменного классификатора для вычисления весов...")
    X_combined = pd.concat([X_train, X_test])
    y_domain = np.array([0] * len(X_train) + [1] * len(X_test))

    import lightgbm as lgb
    domain_classifier = lgb.LGBMClassifier(random_state=SEED, n_jobs=-1)
    domain_classifier.fit(X_combined, y_domain)
    
    train_probs_for_test = domain_classifier.predict_proba(X_train)[:, 1]
    
    weights = train_probs_for_test / (1 - train_probs_for_test + 1e-8)
    weights = weights * (len(X_train) / np.sum(weights))
    
    # --- 4. Обучение модели с весами ---
    print("\nЭтап 1: Обучение модели с вычисленными весами...")
    
    # Создаем сэмплер, который будет использовать веса
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    tokenizer = hydra.utils.instantiate(config.tokenizer)
    train_dataset = OzonDataset(df=train_val_tabular[train_val_tabular['fold'] != 0], transform=transform_train, tokenizer=tokenizer, **config.data)
    val_dataset = OzonDataset(df=train_val_tabular[train_val_tabular['fold'] == 0], transform=transform_train, tokenizer=tokenizer, **config.data)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    tabular_input_dim = len(config.data.tabular_cols)
    model = instantiate(config.arch, tabular_input_dim=tabular_input_dim)
    
    model.load_state_dict(../)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(params=[
        {"params": model.classifier.parameters(), "lr": 5e-4},
        {"params": model.attention.parameters(), "lr": 5e-4},
        {"params": model.image_net.model.features[8].parameters(), "lr": 1e-5},
        {"params": model.image_net.model.features[7].parameters(), "lr": 1e-5},
        {"params": model.text_net.bert.pooler.parameters(), "lr": 1e-5},
        {"params": model.text_net.bert.encoder.layer[10].parameters(), "lr": 1e-5},
        {"params": model.text_net.bert.encoder.layer[11].parameters(), "lr": 3e-6}
    ], weight_decay=1e-4)

    pos_weight = torch.tensor([4.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(10):
        model.train()
        for name, param in model.named_parameters():
                if 'text_net.bert.encoder.layer.10.' in name or \
                'image_net.model.features.7.' in name:
                    param.requires_grad = True
        for batch_data, target in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            image = batch_data['image'].to(device)
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            target = target.to(device).float()
            optimizer.zero_grad()
            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_data, target in tqdm(val_loader, desc=f"Epoch {epoch+1} Valid"):
                image, input_ids, attention_mask = batch_data['image'].to(device), batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device)
                
                logits = model(image, input_ids, attention_mask)
                preds = torch.round(torch.sigmoid(logits))
                
                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

        f1 = f1_score(torch.cat(all_targets), torch.cat(all_preds))
        precision = precision_score(torch.cat(all_targets), torch.cat(all_preds))
        recall = recall_score(torch.cat(all_targets), torch.cat(all_preds))
        accuracy = accuracy_score(torch.cat(all_targets), torch.cat(all_preds))
        print(f"Epoch {epoch+1}, Validation F1-score: {f1:.4f}, Validation accuracy-score: {accuracy:.4f}, Validation precision-score: {precision:.4f}, Validation recall-score: {recall:.4f}")
        logging.info(f"Epoch {epoch+1}, Validation F1-score: {f1:.4f}, Validation accuracy-score: {accuracy:.4f}, Validation precision-score: {precision:.4f}, Validation recall-score: {recall:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            print(f"Saving to best_neural_head.pth")
            torch.save(model.state_dict(), "best_neural_head_v3_with_transfer_v2.pth")
    print("Модель, обученная с весами, сохранена.")

    print("\nЭтап 2: Псевдоразметка тестовых данных...")
    
    # model.load_state_dict(torch.load("model_weighted.pth"))
    model.eval()

    test_dataset = OzonDataset(df=test_tabular, transform=transform_val, tokenizer=tokenizer, **config.data)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    test_probs_list = []
    with torch.no_grad():
        for batch_data, _ in tqdm(test_loader, desc="Получение предсказаний на тесте"):
            # ... отправляем данные на GPU ...
            logits, _ = model(**batch_data)
            probs = torch.sigmoid(logits)
            test_probs_list.append(probs.cpu().numpy())

    test_probs = np.concatenate(test_probs_list)
    
    # Находим "уверенные" предсказания
    high_confidence_mask = (test_probs > 0.95) | (test_probs < 0.05)
    confident_indices = np.where(high_confidence_mask)[0]
    
    X_confident = X_test.iloc[confident_indices]
    y_confident = (test_probs[confident_indices] > 0.5).astype(int)
    
    print(f"Найдено {len(y_confident)} уверенных примеров для псевдоразметки.")
    print(f"Распределение классов в псевдо-метках: \n{pd.Series(y_confident).value_counts(normalize=True)}")
    
    # --- 6. Финальное дообучение ---
    print("\nЭтап 3: Финальное дообучение на объединенных данных...")
    
    # Объединяем исходный трейн и псевдо-трейн
    X_train_augmented = pd.concat([X_train, X_confident])
    y_train_augmented = np.concatenate([y_train, y_confident])
    
    # Создаем новый даталоадер. Теперь можно без весов, просто shuffle=True
    augmented_dataset = OzonDataset()
    augmented_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    
    # Дообучаем модель еще 1-2 эпохи с пониженным learning rate
    # optimizer = RAdam(model.parameters(), lr=1e-5) # <-- Низкий LR
    
    # ... ваш цикл обучения на augmented_loader ...
    
    # Сохраняем финальную, самую лучшую модель
    # torch.save(model.state_dict(), "model_final_semisupervised.pth")
    print("Финальная модель сохранена.")


if __name__ == '__main__':
    train_semi_supervised()
