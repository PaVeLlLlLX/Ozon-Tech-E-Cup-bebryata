import logging
import torch
import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from srcs.utils import instantiate


logger = logging.getLogger('evaluate')

from PIL import Image
import matplotlib.pyplot as plt

# test_path = "/home/pavel/VSC/SHIFT-intensive/data/test/E/E152.jpg"
# image = Image.open(test_path).convert('L')
#image = transform(image).to("cpu")
#alphabet = dataloader.dataset.alphabet
# def show_image(image):
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.show()

@hydra.main(version_base=None, config_path='conf', config_name='evaluate')
def main(config):
    logger.info('Loading checkpoint: {} ...'.format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint, map_location=torch.device('cuda'))

    logger.info(f"Loading test data from {config.data.csv_path}")
    test_df = pd.read_csv(hydra.utils.to_absolute_path(config.data.csv_path))

    # ПРЕДОБРАБОТКА ДАННЫХ КАК В train.py

    data_loader = instantiate(config.data_loader, test_df=test_df)

    tabular_input_dim = len(config.data.tabular_cols)
    model = instantiate(config.arch, tabular_input_dim=tabular_input_dim)
    logger.info(model)

    model.load_state_dict(checkpoint)

    criterion = instantiate(config.loss)
    metrics = [instantiate(met, is_func=True) for met in config.metrics]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    with torch.no_grad():
        for i, (batch_data, target) in enumerate(tqdm(data_loader)):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            target = target.to(device)
            
            output = model(**batch_data)

            loss = criterion(output, target)
            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size

            metric_output = torch.round(output)
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(metric_output.cpu(), target.cpu()) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    logger.info(log)


if __name__ == '__main__':
    main()
