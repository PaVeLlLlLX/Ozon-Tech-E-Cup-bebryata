import logging
import torch
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from srcs.utils import instantiate
from srcs.model.model import MultimodalModel

logger = logging.getLogger('predict')

def predict(config: DictConfig) -> None:
    checkpoint_path = "../ckpt/model_best.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {checkpoint_path}")
        return

    tabular_input_dim = len(config.data.tabular_cols)
    logger.info(f"Tabular input dimension: {tabular_input_dim}")
    
    model = instantiate(config.arch, tabular_input_dim=tabular_input_dim)
    
    model.load_state_dict(checkpoint)
    logger.info("Model loaded successfully.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    logger.info(f"Model moved to {device} and set to evaluation mode.")

    logger.info(f"Loading test data from: {config.data_test.csv_path}")
    test_df_path = hydra.utils.to_absolute_path(config.data_test.csv_path)
    try:
        test_df = pd.read_csv(test_df_path)
    except FileNotFoundError:
        logger.error(f"Test data CSV not found at {test_df_path}")
        logger.error("Please run the preprocessing script for test data first!")
        return

    if config.data.target_col not in test_df.columns:
        logger.warning(f"Target column '{config.data.target_col}' not found in test data. Creating a dummy column.")
        test_df[config.data.target_col] = 0

    test_data_loader = instantiate(config.test_data_loader, test_df=test_df)
    logger.info(f"Test data loader created with {len(test_data_loader.dataset)} samples.")

    predictions = []
    item_ids = []

    with torch.no_grad():
        for batch_data, _ in tqdm(test_data_loader, desc="Predicting"):
            batch_item_ids = batch_data['tabular'][:, 0].cpu().numpy().astype(int)
            item_ids.extend(batch_item_ids)

            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            
            output_logits, _ = model(**batch_data)
            
            batch_preds = torch.round(torch.sigmoid(output_logits)).cpu().numpy()
            predictions.extend(batch_preds)

    logger.info(f"Prediction complete. Total predictions: {len(predictions)}")

    submission_df = pd.DataFrame({
        'id': item_ids,
        'prediction': predictions
    })


    submission_path = 'multimodal_submission.csv'
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission file saved to: {submission_path}")


@hydra.main(version_base=None, config_path='conf/', config_name='train')
def main(config: DictConfig) -> None:
    predict(config)

if __name__ == 'main':
    main()