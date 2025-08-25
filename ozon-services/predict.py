import logging
import torch
import hydra
import pandas as pd
import numpy as np
import json
from omegaconf import DictConfig
from tqdm import tqdm
from srcs.utils import instantiate
from srcs.model.model import MultimodalModel
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


logger = logging.getLogger('predict')

def predict(config: DictConfig) -> None:
    checkpoint_path = "../ckpt/model_best.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {checkpoint_path}")
        return

    tabular_input_dim = len(config.data_test.tabular_cols)
    logger.info(f"Tabular input dimension: {tabular_input_dim}")
    
    model = instantiate(config.arch, tabular_input_dim=tabular_input_dim)
    
    model.load_state_dict(checkpoint)
    logger.info("Model loaded successfully.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    logger.info(f"Model moved to {device} and set to evaluation mode.")

    test_df_path = "../data/processed_by_misha/test.csv"
    try:
        test_df = pd.read_csv(test_df_path)
    except FileNotFoundError:
        logger.error(f"Test data CSV not found at {test_df_path}")
        logger.error("Please run the preprocessing script for test data first!")
        return

    if 'id' not in test_df.columns:
        logger.error("FATAL: 'id' column not found in the processed test.csv.")
        logger.error("Please ensure your preprocessing script saves the 'id' column.")
        return
    
    dataset_df = test_df.copy()
    if config.data.target_col not in dataset_df.columns:
        logger.warning(f"Target column '{config.data.target_col}' not found. Creating a dummy column.")
        dataset_df[config.data.target_col] = 0

    test_data_loader = instantiate(config.test_data_loader, test_df=dataset_df)
    logger.info(f"Test data loader created with {len(test_data_loader.dataset)} samples.")

    # --- 3. Процесс предсказания ---
    all_predictions = []

    with torch.no_grad():
        for batch_data, _ in tqdm(test_data_loader, desc="Predicting"):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            output_logits, _ = model(**batch_data)
            output_preds = (torch.sigmoid(output_logits) > 0.9).long()
            batch_preds = output_preds.cpu().numpy().astype(int)
            # batch_preds = torch.round(torch.sigmoid(output_logits)).cpu().numpy().astype(int)
            all_predictions.extend(batch_preds)

    logger.info(f"Prediction complete. Total predictions: {len(all_predictions)}")

    # --- 4. Формирование и сохранение submission-файла ---
    if len(all_predictions) != len(test_df):
        logger.error(f"Mismatch in lengths: {len(all_predictions)} predictions vs {len(test_df)} rows in test_df.")
        return

    # Добавляем предсказания в исходный DataFrame
    test_df['prediction'] = all_predictions
    
    # ИСПРАВЛЕНИЕ: Выбираем колонку 'id', а не 'ItemID'
    submission_df = test_df[['id', 'prediction']].copy()


    submission_path = 'multimodal_submission_0.9_with_misha_features.csv'
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission file saved to: {submission_path}")


@hydra.main(version_base=None, config_path='conf/', config_name='train')
def main(config: DictConfig) -> None:
    predict(config)

if __name__ == '__main__':
    main()