import os
import sys

import numpy as np
import copy
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import glob
from itertools import product

IMAGE_SIZE = 256
NUM_CLASSES = 10

class_colormap = {  # RGBA information for the different classes
    00: (255, 255, 255, 0),
    10: (250, 150, 10, 255),  # orange (hood)
    20: (20, 100, 20, 255),  # dark green (front door)
    30: (249, 249, 10, 255),  # yellow (rear door)
    40: (10, 248, 250, 255),  # cyan (frame)
    50: (150, 5, 150, 255),  # purple (rear quarter panel)
    60: (10, 250, 10, 255),  # light green (trunk lid)
    70: (20, 20, 250, 255),  # blue (fender)
    80: (250, 10, 250, 255),  # pink (bumper)
    90: (255, 255, 255, 255),  # no color/black (rest of car)
}


def add_transparency_channel(im):
    """This function takes an RGB image and adds a channel after those 3 that
    indicates transparency (scale 0-255, default 255).

    :param im: image with shape (width, height, channels) where the first 3
        channels have the RGB color information
    """

    shape = im.shape
    assert shape[2] >= 3

    transparency_layer = np.full((shape[0], shape[1], 1), 255)
    new_img_array = np.concatenate(
        [im[:, :, :3], transparency_layer, im[:, :, 3:]], axis=2)

    return new_img_array


def make_background_transparent(im, criterion=None, mask_array=None):
    """This function takes an image and for every pixel being a specific color
    (now being black, i.e., equal to (0,0,0)) we set the transparency
    information to 0.

    :param im: car image with shape (width, height, channels) where the first 3
        channels have the RGB color information, and the 4th  is the
        transparency channel"""

    assert im.shape[2] >= 4  # Assert that image has transparency channel

    new_img_array = copy.copy(im)

    if criterion == 'color':
        black_pixels_mask = (new_img_array[..., 0] == 0) & (
                new_img_array[..., 1] == 0) & (new_img_array[..., 2] == 0)
        new_img_array[black_pixels_mask, 3] = 0
    elif criterion is None and mask_array is None:
        background_mask = new_img_array[..., -1] == 0
        new_img_array[background_mask, 3] = 0
    elif mask_array is not None:
        if criterion not in [None, 'mask']:
            print("Warning! You provided a mask, but the criterion parameter"
                  "is not None nor 'mask'. criterion gets set to 'mask' and"
                  "the provided mask is being used.")
        mask = (mask_array[...] == False)
        new_img_array[mask, 3] = 0
    else:
        raise ValueError("Valid values for criterion are 'color', 'class',"
                         "'mask' and None. If you provide a mask, criterion"
                         "gets set to 'mask' and that provided mask will be"
                         "used.")

    return new_img_array


def rescale_to_256(im):
    """Rescale the input image to 256×256 pixels WITHOUT antialiasing"""

    assert im.shape[2] >= 3

    z = copy.copy(im)
    z = np.moveaxis(z, -1, 0)
    z = torch.from_numpy(z)
    z = v2.Resize((256, 256),
                  interpolation=v2.InterpolationMode.NEAREST_EXACT)(z)
    z = z.numpy()
    z = np.moveaxis(z, 0, -1)

    return z


def crop_image(im):
    """This function takes an image and crops away as much as possible while
    only cutting away pixels with transparency information = 0.

    :param im: car image with shape (width, height, channels)  where the first
        3 channels have the RGB color information, and the 4th  is the
        transparency channel"""
    assert im.shape[2] == 5  # Assert that image has transparency channel

    transparency_channel = im[:, :, 3]
    transparency_mask = transparency_channel > 0

    nonzero_coords = np.nonzero(transparency_mask)

    min_row = min(nonzero_coords[0])
    max_row = max(nonzero_coords[0])
    min_col = min(nonzero_coords[1])
    max_col = max(nonzero_coords[1])

    cropped_img_array = im[min_row:max_row + 1, min_col:max_col + 1, :]

    return cropped_img_array


def pad_image(im, size=(256, 256)):
    """This function takes an image and padds it with black pixels with
    transparency = 0 so that the resulting picture has size specified in
    the parameter 'size'.

    :param im: car image with shape (width, height, channels) where the first 3
        channels have the RGB color information, and the 4th  is the
        transparency channel
    :param size: Desired output image size
    """

    shape = im.shape
    assert shape[2] >= 4  # Assert that image has transparency channel
    # Assert that the image is not larger than the desired padded image
    assert all(x <= y for x, y in zip(shape[:1], size))

    padded_image = np.zeros((size[0], size[1], 5), dtype=im.dtype)

    # Calculate the position to paste the cropped image
    paste_x = (size[1] - im.shape[1]) // 2
    paste_y = (size[0] - im.shape[0]) // 2

    # Paste the cropped image onto the padded canvas
    padded_image[paste_y:paste_y + im.shape[0], paste_x:paste_x + im.shape[1],
    :] = im

    return padded_image


def calc_translation(size, image_size=(256, 256), c=0):
    """Calculates how much the car may be translated such that it only a
    portion of 'c' is cut off.

    :param size: size of the car image
    :param image_size: size of the entire image
    :param c: portion of the car that may be cut off
    :return: amount that car may be translated by relative to entire image size
    """

    w, h = size
    x_0, y_0 = image_size
    x_1 = x_0 / 2 - w / 2 + c * w
    y_1 = y_0 / 2 - h / 2 + c * h
    return x_1 / x_0, y_1 / y_0


def affine_transform(im, translate=None, max_degree=12, scale=None,
                     flip_prob=0.5):
    """Applies a number of randomized transformations to the car image, namely
    translation, rotation, scaling, horizontal flipping, and perspective
    modification.

    :param im: car image with shape (width, height, channels) where the first 3
        channels have the RGB color information, and the 4th is the
        transparency channel
    :param translate: tuple of maximum absolute fraction for horizontal and
        vertical translations
    :param max_degree: Range [-max_degree, +max_degree] of degrees to select
        from
    :param scale: scaling factor interval, e.g (a, b), then scale is randomly
        sampled from the range a <= scale <= b
    :param flip_prob: probability that image will be flipped horizontally
    :return: image with all randomized transformations applied
    """

    assert im.shape[2] >= 4  # Assert that image has transparency channel

    a = postprocessing(im, brightness=0.0, contrast=0.0, saturation=0.0,
                       hue=0.0, kernel_size=(1, 1))

    a = np.moveaxis(a, -1, 0)
    a = torch.from_numpy(a)
    a = v2.RandomPerspective(distortion_scale=0.2, p=1.0,
                             interpolation=v2.InterpolationMode.NEAREST)(a)
    a = v2.RandomAffine([-max_degree, max_degree], translate=translate,
                        scale=scale,
                        interpolation=v2.InterpolationMode.NEAREST)(a)
    a = v2.RandomHorizontalFlip(p=flip_prob)(a)
    a = a.numpy()
    a = np.moveaxis(a, 0, -1)

    return a


def place_foreground_on_background_pic(im, bim):
    """Place the car picture on top of the background image, respecting the
    transparency information of the car image.

    :param im: foreground image with shape (width, height, channels) where the
        first 3 channels have the RGB color information, and the 4th  is the
        transparency channel
    :param bim: background image with shape (width, height, channels) where
        the first 3 channels have the RGB color information
    :return: merged image
    """

    assert im.shape[2] >= 4  # Assert foreground image has alpha channel
    assert bim.shape[2] >= 3  # Assert background image has RGB channels

    # Assert that the pictures have the same dimension
    assert im.shape[:2] == bim.shape[:2]
    width, height = im.shape[:2]
    im_num_channels, bim_num_channels = im.shape[2], bim.shape[2]

    # Extract the RGB and Alpha channels from the car image
    foreground_rgb = im[:, :, :3]
    foreground_alpha = im[:, :, 3] / 255.0
    foreground_alpha_complement = 1 - foreground_alpha

    # Blend the RGB and Alpha channels using the car's Alpha channel
    blended_rgb = foreground_rgb * foreground_alpha[:, :, np.newaxis]
    blended_rgb += bim[:, :, :3] * foreground_alpha_complement[:, :,
                                   np.newaxis]

    # Create a new image with RGBA channels
    if im_num_channels >= bim_num_channels:
        new_img_array = copy.copy(im)
    else:
        new_img_array = copy.copy(bim)
    new_img_array[:, :, :3] = blended_rgb
    # Set the Alpha channel from the car image
    new_img_array[:, :, 3] = np.full_like(im[:, :, 3], 255)
    # Keep the class channel from the car image

    return new_img_array


def remove_transparency_channel(im):
    """This function takes an RGBA image (RGB with transparency information)
    and removes the transparency channel.

    :param im: image with shape (width, height, channels) where the first 3
        channels have the RGB color information, and the 4th  is the
        transparency channel
    :return: image without transparency channel
    """

    shape = im.shape
    assert shape[2] >= 4  # Assert that image has transparency channel

    new_img_array = np.concatenate([im[:, :, :3], im[:, :, -1:]], axis=2)
    # Assert that image has no transparency channel
    assert new_img_array.shape[2] == 4

    return new_img_array


def postprocessing(im, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0,
                   kernel_size=(3, 3), sigma=(0.8, 0.8)):
    """Applies a number of randomized transformations to the final image,
    namely color jittering and gaussian blur.

    :param im: image with shape (width, height, channels) where the first 3
        channels have the RGB color information
    :param brightness: brightness randomized color jittering amount
    :param contrast: contrast randomized color jittering amount
    :param saturation: saturation randomized color jittering amount
    :param hue: hue randomized color jittering amount
    :param kernel_size: kernel size of randomized gaussian blur
    :param sigma: sigma of randomized gaussian blur
    :return: image with all randomized transformatins applied
    """

    a = im[:, :, :3]
    a = np.moveaxis(a, -1, 0)
    a = torch.from_numpy(a).float()
    a = v2.ColorJitter(brightness=brightness, contrast=contrast,
                       saturation=saturation, hue=hue)(a)
    a = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(a)
    a = a.numpy().astype(int)
    a = np.moveaxis(a, 0, -1)
    a = np.concatenate([a, im[:, :, 3:]], axis=2)

    return a


def get_most_likely_class(channels):
    """Returns the most likely class for every pixel

    :param channels: shape 9x256x256. Element (i, x, y) indicates the
        probability that pixel (x,y) belongs to i'th class
    :return: For each pixel (x,y), return most likely class
    """

    # assert channels.shape == (NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)

    max_channel = np.argmax(channels, axis=0)  # output shape (256,256)
    classes = 10 * (max_channel)

    return classes


def clean_class_channels(im):
    """This function cleans up the f"""

    classes = im[:, :, -1]
    classes_snapped = (np.round(np.divide(classes, 10), decimals=0) * 10)
    im[:, :, -1] = classes_snapped

    return im.astype(int)


def color_image(input_classes):
    """This function creates an RGBA image where the pixels are colored
    according to its class.

    :param input_classes: array indicating classes. May be shaped
        (10, 256, 256) where for each pixel (i,j), the probabilities in
        input_classes[:, i, j] indicate which class that pixel should belong
        to. Can also be shaped (256, 256) where in each pixel there is a class
        10, ..., 90. Former array (shaped (10, 256, 256)) gets automatically
        converted to the latter (shaped (256, 256)) using
        get_most_likely_class().
    :return: RGBA image, pixels colored according to class_colormap.
    """

    shape = input_classes.shape

    # Test if we got a 10×256×256 class array with probabilities, or just a
    # 256×256 array with classes
    # if shape == (NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE):
    if len(shape) == 3:
        new_input_classes = get_most_likely_class(input_classes)
    # elif shape == (IMAGE_SIZE, IMAGE_SIZE):
    elif len(shape) == 2:
        new_input_classes = copy.copy(input_classes)
    else:
        raise ValueError(f'You must input either a tensor with class \
probabilities shaped ({NUM_CLASSES}, {IMAGE_SIZE}, {IMAGE_SIZE}), or an array \
indicating what class the pixel belongs to shaped ({IMAGE_SIZE}, {IMAGE_SIZE}\
), but you input {shape}.')
        pass

    # Make RGBA image
    height, width = new_input_classes.shape
    colored_image = np.zeros((height, width, 4), dtype=np.uint8)

    for i, rgba in class_colormap.items():
        class_mask = new_input_classes[...] == i
        colored_image[class_mask] = rgba

    return colored_image


def color_classes(img):
    """This function takes an image with either 4 channels (RGBC,
    red-green-blue-class) or 5 channels (RGBAC, red-green-blue-alpha-class)
    and colors the image according to the classes.

    :param img: Input image.
    :return: Image colored according to class.
    """

    c = color_image(img[:, :, -1])
    new_img = place_foreground_on_background_pic(c, img)
    return new_img


def main(im, bim, verbose=False, plotting=False, image_size=256,
         max_crop_margin=0, max_degree=12, scaling=(0.8, 1.2), mask=None):
    """Executes the entire augmentation step.

    :param im: car image with shape (width, height, channels) where the first
        3 channels have the RGB color information
    :param bim: background image with shape (width, height, channels) where
        the first 3  channels have the RGB color information
    :param image_size: dimension of the entire image
    :param max_crop_margin: ∈ [0,1]. Indicates how much of the car can be
        cropped off when doing a random translation in affine_transform(). It
        is still possible that more is cropped off, e.g. when the car gets
        moved in a corner, gets rotated, or in the scaling step.
    :param max_degree: ∈ [0,∞]. Indicates how much the car may get rotated.
        Random sample from [-MAX_DEGREE,MAX_DEGREE].
    :param scaling: ∈ [0,∞]². Indicates how much the car may get scaled. Random
        sample from [min,max].
    :param mask: Binary mask array of shape (256, 256) which indicates what
        pixels belong to the car. If this is provided, we crop the car
        according to this array.
    :return:
    """

    # Copy image array objects so that we can modify them
    img = copy.copy(im)
    bimg = copy.copy(bim)

    # Create dict to store the intermediate pictures if verbose or plotting is
    # enabled
    if plotting or verbose:
        images_dict = {}

    # 1: Take car image and add transparency information channel
    img = add_transparency_channel(img)
    bimg = add_transparency_channel(bimg)
    if plotting or verbose:
        images_dict['Car'] = img[:, :, :3]
        images_dict['Car w/ Class'] = color_classes(img)[:, :, :3]
        images_dict['Background'] = bimg

    # 2: Make background transparent
    img = make_background_transparent(img, mask_array=mask)
    if plotting or verbose:
        images_dict['Car w/ Transparency'] = img[:, :, :4]
        images_dict['Car w/ Transparency & Class'] = \
            color_classes(img)[:, :, :4]

    # 3: Move car to center; 3.1: Crop to smallest rectangle still containing
    # the car
    img = crop_image(img)
    if plotting or verbose:
        images_dict['Cropped Car w/ Transparency'] = img[:, :, :4]
        images_dict['Cropped Car w/ Transparency & Class'] = \
            color_classes(img)[:, :, :4]

    # 3: Move car to center; 3.2: Place it in the middle of a transparent
    # 256×256 image
    car_shape = img.shape[:2]
    img = pad_image(img, size=(image_size, image_size))
    if plotting or verbose:
        images_dict['Padded Car w/ Transparency'] = img[:, :, :4]
        images_dict['Padded Car w/ Transparency & Class'] = \
            color_classes(img)[:, :, :4]

    # 4: Do affine transformations, i.e., rotate, translate, scale, flip, and
    # perspective change.
    translation = calc_translation(car_shape, (image_size, image_size),
                                   max_crop_margin)
    img = affine_transform(img, translate=translation,
                           max_degree=max_degree, scale=scaling)
    if plotting or verbose:
        title = 'Translated, Rotated, Scaled Car,\nFlip, Perspective w/ Transparency'
        images_dict[title] = img[:, :, :4]
        title = 'Translated, Rotated, Scaled Car,\nFlip, Perspective w/ Transparency & Class'
        images_dict[title] = color_classes(img)[:, :, :4]

    # 5: Place car image on top of the background image
    img = place_foreground_on_background_pic(img, bimg)
    if plotting or verbose:
        images_dict['Merged with background'] = img[:, :, :4]
        images_dict['Merged with background w/ Class'] = \
            color_classes(img)[:, :, :4]

    # 6: Remove transparency information from picture
    img = remove_transparency_channel(img)

    # 7: Do color jittering, blur
    img = postprocessing(img)
    if plotting or verbose:
        images_dict['Color Jitter and Blur'] = img[:, :, :3]
        images_dict['Color Jitter and Blur w/ Class'] = \
            color_classes(img)[:, :, :3]

    if plotting:
        n_rows = 4
        n_cols = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
        fig.subplots_adjust(hspace=0.5)

        for i, (key, val) in enumerate(images_dict.items()):
            axes.ravel()[i].imshow(val)
            axes.ravel()[i].set_title(key)

        for ax in axes.ravel():
            # ax.axis('off')
            ax.tick_params(left=False, right=False, labelleft=False,
                           labelbottom=False, bottom=False)

        plt.show()

    if verbose:
        return images_dict

    return img


def load_background(path):
    bimg = imread(path)
    bimg = resize(bimg, (256, 256), anti_aliasing=True)
    bimg = bimg * 255
    # bimg = v2.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=v2.InterpolationMode.BILINEAR)(bimg)
    bimg_array = np.array(bimg).astype(int)
    if len(bimg_array.shape) == 2:
        bimg_array = np.stack([bimg_array] * 3, axis=2)

    return bimg_array


if __name__ == '__main__':
    # Load car image
    img_path = '../carseg_data/arrays_v2/orange_3_doors_0001.npy'
    img_array = np.load(img_path)

    # Load car mask determined by detectron2
    mask_path = '../carseg_data/arrays_car_masks/orange_3_doors_0001.npy'
    mask_array = np.load(mask_path)

    # Load background image and scale it to 256×256
    bimg_path = '../carseg_data/landscapes/0001.jpg'
    bimg_array = load_background(bimg_path)

    s="""img_array = main(img_array, bimg_array, mask=mask_array,
                     plotting=False)
    plt.imshow(img_array[:, :, :3])
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    # plt.show()
    # cwd = os.getcwd()
    """
    cars_list = glob.glob('../carseg_data/arrays_v2/photo_*.npy') #+ \
                #glob.glob('../carseg_data/arrays_v2/orange_3_doors_*.npy')
    backgrounds_list = glob.glob('../carseg_data/landscapes/*.jpg')
    array_path_list = list(product(cars_list, backgrounds_list))

    if not array_path_list:
        raise ValueError('You are in the wrong directory. Please cd into the helper folder and then execute this file.')

    # Filter out every
    num_images = 1  # 10_000
    l = len(array_path_list)
    if num_images > l:
        raise ValueError(f'You have {l} images, but you want {num_images}.')
    filter = int(l / num_images)
    array_path_list = array_path_list[::filter][:num_images]

    i = 0

    for (car_file_path, background_file_path) in array_path_list:
        print(f'i={i:06d}')

        # if i <= 1411:
        #     i += 1
        #     continue

        # Load car image
        car_array = np.load(car_file_path)

        # Load car mask determined by detectron2
        mask_path = car_file_path
        mask_path = mask_path.replace('/arrays_v1/', '/arrays_car_masks/')
        mask_path = mask_path.replace('/arrays_v2/', '/arrays_car_masks/')
        mask_path = mask_path.replace('/arrays_v3/', '/arrays_car_masks/')
        mask_path = mask_path.replace('/arrays/', '/arrays_car_masks/')
        mask_array = np.load(mask_path)

        # Load background image and scale it to 256×256. Add
        background_array = load_background(background_file_path)

        img_array = main(car_array, background_array, scaling=(1.1, 1.2), max_degree=45)
        img_array = img_array.astype(np.uint8)

        np.save(f'../carseg_data/arrays_augmented/{i:06d}.npy', img_array)

        if i % 1_000 == 0:
            plt.imshow(img_array[:, :, :3])
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            plt.show()

        i += 1
