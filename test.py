import argparse
import platform
import torch
import torchvision

torchvision.disable_beta_transforms_warning()
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from helper.augment_data import color_image
from PIL import Image



def convert_png_transparent(image, bg_color=(255, 255, 255)):
    image = image.convert("RGBA")
    array = np.array(image, dtype=np.ubyte)
    mask = (array[:, :, :3] == bg_color).all(axis=2)
    alpha = np.where(mask, 0, 255)
    array[:, :, -1] = alpha
    return Image.fromarray(np.ubyte(array))


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = config.init_obj('loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # windows path fix
    if (platform.system() == 'Windows'):
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        checkpoint = torch.load(config.resume, map_location='cpu')
    else:
        if (torch.cuda.is_available()):
            checkpoint = torch.load(config.resume)
        else:
            checkpoint = torch.load(config.resume, map_location='cpu')
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    metricPerImage = []

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            foreground_output = color_image(output[0].cpu().detach().numpy())
            foreground_target = color_image(target[0].cpu().detach().numpy())
            background = convert_png_transparent(
                Image.fromarray(np.load(f"./test_set/photo_{i + 1:04d}.npy"))).convert("RGBA")
            background_target = convert_png_transparent(
                Image.fromarray(np.load(f"./test_set/photo_{i + 1:04d}.npy"))).convert("RGBA")
            foreground_output = convert_png_transparent(Image.fromarray(foreground_output))
            foreground_target = convert_png_transparent(Image.fromarray(foreground_target))

            background.paste(foreground_output, (0, 0), foreground_output)
            background_target.paste(foreground_target, (0, 0), foreground_target)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Output and Target of Test sample ' + str(i))
            ax1.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            ax2.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            ax1.imshow(background)
            ax2.imshow(background_target)

            plt.savefig(f'/content/car_parts_segmentation/images/test/{str(i)}')
            plt.close()

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

            metricPerImage.append(module_metric.car_part_accuracy(output, target))

            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    plt.hist(metricPerImage)
    plt.title("Histogram of car part accuracy per real photo")
    plt.xlabel("Car part accuracy")
    plt.ylabel("Number of images")
    # plt.show()
    plt.savefig("hist.pdf")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
