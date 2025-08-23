import os
import yaml
import logging
from functools import partial, update_wrapper
from importlib import import_module
from itertools import repeat
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_logger(name=None,state='train'):
    if is_master():
        hydra_conf = OmegaConf.load(f'run/{state}/.hydra/hydra.yaml')
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return logging.getLogger(name)


def collect(scalar):
    """
    util function for DDP.
    syncronize a python scalar or pytorch scalar tensor between GPU processes.
    """
    # move data to current device
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar)
    scalar = scalar.to(dist.get_rank())

    # average value between devices
    dist.reduce(scalar, 0, dist.ReduceOp.SUM)
    return scalar.item() / dist.get_world_size()

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def instantiate(config, *args, is_func=False, **kwargs):
    """
    wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert '_target_' in config, f'Config should have \'_target_\' for class instantiation.'
    target = config['_target_']
    if target is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = target.rsplit('.', 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # make partial function with arguments given in config, code
        kwargs.update({k: v for k, v in config.items() if k != '_target_'})
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, **kwargs)

def write_yaml(content, fname):
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=2, sort_keys=False)

def write_conf(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = OmegaConf.to_container(config, resolve=True)
    write_yaml(config_dict, save_path)

def get_logits(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    logits_list = []
    targets_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            logits_list.append(logits)
            targets_list.append(labels)

    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    logits = logits.cpu()
    targets = targets.cpu()
    return logits, targets

def delete_rows_without_images(df: pd.DataFrame, images_path: str, item_id_col: str = 'ItemID') -> pd.DataFrame:
    """
    Фильтрует DataFrame, оставляя только те строки, для которых существуют
    соответствующие изображения в указанной папке.

    Args:
        df (pd.DataFrame): Исходный DataFrame для фильтрации.
        images_path (str): Путь к директории с изображениями.
        item_id_col (str): Название колонки с ID элемента (по умолчанию 'ItemID').

    Returns:
        pd.DataFrame: Отфильтрованный DataFrame.
    """
    print(f"Проверка наличия изображений: {images_path}...")

    if not os.path.isdir(images_path):
        print(f"Директория '{images_path}' не найдена.")
        return df
    try:
        available_image_ids = {os.path.splitext(f)[0] for f in os.listdir(images_path)}
        if not available_image_ids:
             print(f"В директории '{images_path}' не найдено изображений.")
             return df
    except Exception as e:
        print(f"Ошибка при чтении директории '{images_path}': {e}.")
        return df

    if item_id_col not in df.columns:
        print(f"Колонка '{item_id_col}' не найдена в DataFrame.")
        return df
        
    initial_rows = len(df)

    df_filtered = df[df[item_id_col].astype(str).isin(available_image_ids)].copy()
    
    final_rows = len(df_filtered)
    removed_count = initial_rows - final_rows
    
    print("Проверка завершена.")
    print(f"  Исходное количество строк: {initial_rows}")
    print(f"  Удалено строк без изображений: {removed_count}")
    print(f"  Итоговое количество строк: {final_rows}")
    
    return df_filtered
