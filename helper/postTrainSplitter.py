import glob
from itertools import product
import os
import shutil
from matplotlib import pyplot as plt

import numpy as np

from augment_data import load_background, main

split = 0.2

outdir = "../carseg_data/"
photoDir = "../carseg_data/photo_array"

background_file_path = "../carseg_data/images/landscapes"





testdir = os.path.join(outdir, "postTrain/test")
traindir = os.path.join(outdir, "postTrain/train")


os.makedirs(testdir, exist_ok=True)
os.makedirs(traindir, exist_ok=True)

numphoto = len(os.listdir(photoDir))

numtestphoto = int(numphoto*split)

for i in range(numtestphoto):
    photo = f"photo_{i+1:04d}.npy"
    shutil.copyfile(os.path.join(photoDir, photo), os.path.join(testdir, photo))

for i in range(numtestphoto, numphoto):
    photo = f"photo_{i+1:04d}.npy"
    shutil.copyfile(os.path.join(photoDir, photo), os.path.join(traindir, photo))



cars_list = glob.glob(traindir+'/*.npy')
backgrounds_list = glob.glob(background_file_path+'/*.jpg')
array_path_list = list(product(cars_list, backgrounds_list))

if not array_path_list:
    raise ValueError('You are in the wrong directory. Please cd into the helper folder and then execute this file.')


# Filter out every
num_images = 1000
l = len(array_path_list)
filter = int(l/num_images)
array_path_list = array_path_list[::filter][:num_images]

i = 0

for (car_file_path, background_file_path) in array_path_list:
    print(f'{i=:06d}')
    # Load car image
    car_array = np.load(car_file_path)

    # Load car mask determined by detectron2
    #mask_path = car_file_path.replace('arrays', 'arrays_car_masks')
    mask_array = (np.load(car_file_path)[:, :, 3] != 0)

    # Load background image and scale it to 256×256. Add
    background_array = load_background(background_file_path)

    print(car_array.shape)
    print(background_array.shape)
    print(mask_array.shape)
    img_array = main(car_array, background_array, mask=mask_array)
    img_array.astype(np.uint8)

    np.save(f'{traindir}/{i:06d}.npy', img_array)

    if i % 1_000 == 0:
        plt.imshow(img_array[:, :, :3])
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.show()

    i += 1