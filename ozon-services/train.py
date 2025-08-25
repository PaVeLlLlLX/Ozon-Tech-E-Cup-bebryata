import numpy as np
import hydra
import torch
import torchvision
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from srcs.trainer import Trainer
from srcs.utils import instantiate, get_logger, is_master
from srcs.model.model import get_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

logging.basicConfig(filename="stage_1_nn.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

def train_worker(config: DictConfig):
    logger = get_logger('train')
    if config.cwd and is_master():
        logger.info(f"Working directory: {config.cwd}")
        print(OmegaConf.to_yaml(config))

    logger.info("Loading and preparing data...")
    raw_df = pd.read_csv(hydra.utils.to_absolute_path(config.data.csv_path))
    #raw_df.info()
    #logger.info(f"Loading processed tabular data from: {config.data.processed_tabular_path}")
    # processed_df = pd.read_parquet(
    #     hydra.utils.to_absolute_path(config.data.processed_tabular_path)
    # )

    # Объединяем датафреймы по 'id'
    # Убедимся, что 'id' есть в обоих датафреймах для мержа
    # if 'id' not in raw_df.columns or 'id' not in processed_df.columns:
    #     raise ValueError("'id' column must be present in both raw and processed dataframes for merging.")
        
    # Сохраняем из сырого датафрейма только id и колонки для NLP/CV
    # Это предотвратит дублирование колонок
    # nlp_cv_cols = config.data.text_cols + [config.data.id_col, 'id'] # Добавляем ItemID и id для связки
    # raw_df_subset = raw_df[nlp_cv_cols].drop_duplicates()

    # Объединяем по 'id'. В full_df теперь будут и сырые текстовые, и обработанные табличные данные
    # full_df = pd.merge(raw_df_subset, processed_df, on='id', how='inner')
    
    # logger.info(f"Final merged dataframe shape: {full_df.shape}")

    # ПРЕДОБРАБОТКА ДАННЫХ
    
    train_df, val_df = train_test_split(
        raw_df,
        test_size=config.data.val_size,
        random_state=SEED,
        stratify=raw_df[config.data.target_col]
    )
    logger.info(f"Data split: {len(train_df)} train, {len(val_df)} validation samples.")

    data_loader, valid_data_loader = instantiate(
        config.data_loader, 
        train_df=train_df, 
        val_df=val_df
    )

    tabular_input_dim = len(config.data.tabular_cols)
    logger.info(f"Tabular input dimension: {tabular_input_dim}")
    model = instantiate(config.arch, tabular_input_dim=tabular_input_dim)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(model)
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')
    
    criterion = instantiate(config.loss)
    metrics = [instantiate(met, is_func=True) for met in config['metrics']]
    
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)
        
    optimizer = torch.optim.AdamW([
    {"params": model.parameters()}
    ], lr=1e-3, weight_decay=1e-4)

    trainer = Trainer(model, 3, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler
                      )
    trainer.train()
    
    for name, param in model.named_parameters():
        if 'text_net.bert.encoder.layer.11.' in name or \
            'text_net.bert.pooler.' in name or \
            'image_net.model.features.8.' in name:
            param.requires_grad = True


    optimizer = torch.optim.AdamW([
        {"params": model.proj.parameters(), "lr": 5e-4},
        {"params": model.tabnet.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 5e-4},
        {"params": model.cross_attention.parameters(), "lr": 5e-5},
        {"params": model.image_net.model.features[8].parameters(), "lr": 2e-5},
        {"params": model.text_net.bert.pooler.parameters(), "lr": 2e-5},
        {"params": model.text_net.bert.encoder.layer[11].parameters(), "lr": 2e-5}
    ], weight_decay=1e-4)
    
    trainer = Trainer(model, 6, criterion, metrics, optimizer,
                       config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler
                      )
    
    trainer.train()

    for name, param in model.named_parameters():
        if 'text_net.bert.encoder.layer.10.' in name or \
            'image_net.model.features.7.' in name:
            param.requires_grad = True
    
    optimizer = torch.optim.AdamW(params=[
        {"params": model.proj.parameters(), "lr": 1e-4},
        {"params": model.tabnet.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 2e-4},
        {"params": model.cross_attention.parameters(), "lr": 2e-5},
        {"params": model.image_net.model.features[8].parameters(), "lr": 1e-5},
        {"params": model.image_net.model.features[7].parameters(), "lr": 1e-5},
        {"params": model.text_net.bert.pooler.parameters(), "lr": 3e-6},
        {"params": model.text_net.bert.encoder.layer[10].parameters(), "lr": 1e-5},
        {"params": model.text_net.bert.encoder.layer[11].parameters(), "lr": 1e-6}
    ], weight_decay=1e-4)
    
    trainer = Trainer(model, 6, criterion, metrics, optimizer,
                       config=config,
                       data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                       lr_scheduler=lr_scheduler
                       )
    
    trainer.train()
    # try:
    #     trainer.train()
    # except Exception as e:
    #   logging.error(msg=f"{e}")

def init_worker(working_dir, config):
    # initialize training config
    config = OmegaConf.create(config)
    config.cwd = working_dir
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)
    # start training processes
    train_worker(config)


@hydra.main(version_base=None, config_path='conf/', config_name='train')
def main(config):
    n_gpu = torch.cuda.device_count()
    if config.gpu:
        config['n_gpu'] = n_gpu
    else:
        config['n_gpu'] = 0

    working_dir = str(Path.cwd().relative_to(hydra.utils.get_original_cwd()))
    if config.resume is not None:
        config.resume = hydra.utils.to_absolute_path(config.resume)
    config = OmegaConf.to_yaml(config, resolve=True)
    init_worker(working_dir, config)

if __name__ == '__main__':
    main()