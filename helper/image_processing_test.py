import numpy as np
import matplotlib.pyplot as plt
from helper.image_processing import color_image

# function test
input_pic = [[[1, 1, 0],
              [0, 0, 0],
              [0, 0, 0]],

             [[0, 0, 0],
              [1, 1, 0],
              [0, 0, 0]],

             [[0, 0, 0],
              [0, 0, 0],
              [1, 1, 0]],

             [[0, 0, 1],
              [0, 0, 1],
              [0, 0, 1]]]

c = color_image(np.array(input_pic))
plt.imshow(c)
plt.show()
