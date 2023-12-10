import argparse
import platform
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from imgconvert import array_to_image
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from helper.augment_data import color_image
import cv2

def resizeImage(image, desired_size = 256):
        old_size = image.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format

        image = cv2.resize(image, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = 0, delta_h
        left, right = 0, delta_w

        color = [0, 0, 0]
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

def main(config):
    model = config.init_obj('arch', module_arch)
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
    
    
    image = cv2.imread("./test_data/real/0100.jpg")
    image = torch.tensor(resizeImage(image).transpose(2,0,1)/255,dtype=torch.float32).unsqueeze(0)
    
    
    
    image = image.to(device)
    target = model(image)
    target = target.squeeze(0)
    target =  torch.argmax(target, dim=0)
    target = target.detach().numpy()
    target = target*10
    target = np.asarray(target,dtype=np.uint8)
    target = array_to_image(target)
    target = np.asarray(target,dtype=np.float32)
    
    image = image.squeeze(0)
    image = np.asarray(image).transpose(1,2,0)*255
    
    print(image.shape)
    print(target.shape)
    
    dst = cv2.addWeighted(image, 1, target , 1, 0, dtype=cv2.CV_8U)
    cv2.imshow("image",dst)
    cv2.waitKey(0)
    
    
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