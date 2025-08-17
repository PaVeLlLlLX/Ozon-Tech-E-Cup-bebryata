import logging
import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from srcs.utils import instantiate


logger = logging.getLogger('evaluate')

from PIL import Image
import matplotlib.pyplot as plt

test_path = "/home/pavel/VSC/SHIFT-intensive/data/test/E/E152.jpg"
image = Image.open(test_path).convert('L')
#image = transform(image).to("cpu")
#alphabet = dataloader.dataset.alphabet
def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

@hydra.main(version_base=None,config_path='conf', config_name='evaluate')
def main(config):
    logger.info('Loading checkpoint: {} ...'.format(config.checkpoint))
    #checkpoint = torch.load(config.checkpoint)
    checkpoint = torch.load('/home/pavel/VSC/SHIFT-intensive/ckpt/model_best.pth', map_location=torch.device('cpu'))
    # setup data_loader instances
    print(config.data_loader)
    data_loader = instantiate(config.data_loader)

    # restore network architecture
    model = instantiate(config.arch)
    logger.info(model)

    # load trained weights
    model.load_state_dict(checkpoint)

    # instantiate loss and metrics
    criterion = instantiate(config.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in config.metrics]

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            #show_image(data)
            print("OUTPUT", output, target)
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            metric_output = output.argmax(dim=1)

            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(metric_output.cpu(), target.cpu()) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    logger.info(log)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
