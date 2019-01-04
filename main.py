import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import sys
import matplotlib.pyplot as plt
from data_aug import tf_resize_images, central_scale_images,flip_images
import cv2
# create image paths list
IMAGE_SIZE = 256
CH = 3
image_paths = []
image_paths.append('sample/normal/000001.png')

# store returned list of numpy arrays
resize_imgs = tf_resize_images(image_paths,IMAGE_SIZE,CH)
scaled_imgs = central_scale_images(resize_imgs, [0.90, 0.75, 0.60],IMAGE_SIZE,CH)
flipped_images = flip_images(resize_imgs,IMAGE_SIZE,CH)

# iterate through the array, and save the image to the directory
# data is array of numpy arrays
# so data[i] used to write, not data
for i in range(len(resize_imgs)):
    #img = cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR)
    plt.imshow(flipped_images[i])
    plt.show()
    #cv2.imwrite("image_output" + str(i) + ".jpg", img)
    # unable to show the image output properly - to-do
# data = np.array(data)