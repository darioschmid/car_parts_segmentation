import numpy as np

bgr_values = {
    10: (10, 150, 250),  # orange
    20: (20, 100, 20),  # dark green
    30: (10, 249, 249),  # yellow
    40: (250, 248, 10),  # cyan
    50: (150, 5, 150),  # purple
    60: (10, 250, 10),  # light green
    70: (250, 20, 20),  # blue
    80: (250, 10, 250),  # pink
    90: (100, 100, 100),
    0: (0, 0, 0),# no color
}

def _distance(c1, c2):
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    return np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def classify_color(b, g, r):
    distances = [_distance((b, g, r), c) for c in bgr_values.values()]
    argmin_distance = np.argmin(distances)
    argmin_class = list(bgr_values.keys())[argmin_distance]
    return argmin_class


def image_to_array(image):
    seg_image = np.zeros((image.shape[0], image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i, j]
            seg_image[i, j] = classify_color(b, g, r)
    return seg_image

def array_to_image(array):
    image = np.zeros((array.shape[0], array.shape[1],3),np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            image[i, j] = np.asarray(bgr_values[array[i, j]],np.uint8)
            

    return np.asarray(image,np.uint8)



if __name__ == "__main__":
    from skimage.io import imread

    array_path = "carseg_data/arrays/photo_0001.npy"
    array = np.array(np.load(array_path))

    image_path = "carseg_data/images/photo/with_segmentation/0001.jpg"
    image = imread(image_path)
    print(array)
    print(f"{array.shape=}")
