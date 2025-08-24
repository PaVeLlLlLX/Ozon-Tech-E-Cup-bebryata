import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import hydra
import os
from omegaconf import DictConfig
from stage_1_nn import NeuralFeatureExtractor
from srcs.model.text_model import TextNet
from srcs.model.image_model import ImageNet
from srcs.data_loader.data_loaders import OzonDataset
from torch.utils.data import DataLoader
from srcs.utils import instantiate, get_logger, is_master
from sklearn.model_selection import train_test_split

from srcs.data_loader.data_loaders import OzonDataset, transform_val
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'




@hydra.main(version_base=None, config_path='conf/', config_name='train')
def extract_features(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    model = NeuralFeatureExtractor().to(device)
    model.load_state_dict(torch.load("best_neural_head.pth"))
    model.eval()

    # df = pd.read_csv(hydra.utils.to_absolute_path(config.data.csv_path))
    # train_df, val_df = train_test_split(df, test_size=config.data.val_size, random_state=SEED, stratify=df[config.data.target_col])
    # df = pd.read_csv("../data/train_with_folds.csv")
    
    VAL_FOLD = 0
    df_train_val = pd.read_csv("../data/train_with_folds.csv", index_col='id')

    df_test = pd.read_csv("../data/processed/test.csv", index_col='id')

    tokenizer = hydra.utils.instantiate(config.tokenizer)
    
    datasets = {
        'train': OzonDataset(df=df_train_val[df_train_val['fold'] != 0], transform=transform_val, tokenizer=tokenizer, **config.data),
        'val': OzonDataset(df=df_train_val[df_train_val['fold'] == 0], transform=transform_val, tokenizer=tokenizer, **config.data),
        'test': OzonDataset(df=df_test, transform=transform_val, tokenizer=tokenizer, **config.data_test)
    }
    
    for name, dataset in datasets.items():
        print(f"Processing {name} set...")
        loader = DataLoader(dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0)
        
        all_features = []
        all_ids = []

        with torch.no_grad():
            for batch_data, _ in tqdm(loader, desc=f"Extracting features for {name}"):
                image, input_ids, attention_mask = batch_data['image'].to(device), batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device)
                item_ids = batch_data["id"].tolist()

                text_out = model.text_net(input_ids, attention_mask)
                img_out = model.image_net(image)
            
                features = torch.cat([text_out, img_out], dim=1)
                
                all_features.append(features.cpu().numpy())
                all_ids.extend(item_ids)

        final_features = np.vstack(all_features)
        feature_df = pd.DataFrame(final_features, index=all_ids)
        feature_df.columns = [f"neural_{i}" for i in range(final_features.shape[1])]
        
        output_path = f"{name}_neural_features.parquet"
        feature_df.to_parquet(output_path)
        print(f"Признаки для {name} сохранены в {output_path}")

if __name__ == '__main__':
    extract_features()