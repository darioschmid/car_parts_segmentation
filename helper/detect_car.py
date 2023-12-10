import detectron2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from PIL import Image
import glob
import os

setup_logger()

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


def show_image(img):
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def save_img(array, filename):
    image_obj = Image.fromarray(array)
    image_obj.save(filename)


# Setup
cfg = detectron2.config.get_cfg()
# Use CPU
cfg.MODEL.DEVICE = 'cpu'
# add project-specific config (e.g., TensorMask) here if you're not running a
# model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the
# https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

files_list = glob.glob('../carseg_data/arrays/black_5_doors_*.npy') + \
             glob.glob('../carseg_data/arrays/orange_3_doors_*.npy')
for file_path in files_list:
    file_name = os.path.basename(file_path)
    print(f'Processing file {file_name}')
    im = np.load(file_path)[:, :, :3]
    outputs = predictor(im[:, :, ::-1])

    v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    save_img(out.get_image(), '../images/detect_car_output/' +
             file_name.split('.')[0] + '.png')
    # show_image(out.get_image())

    masks = outputs["instances"].pred_masks
    mask = np.any(masks.numpy(), axis=0)

    np.save('../carseg_data/arrays_car_masks/' + file_name, mask)
    # break
