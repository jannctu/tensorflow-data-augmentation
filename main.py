import tensorflow as tf
#import matplotlib.image as mpimg
from scipy.misc import imread
import numpy as np
import sys
import matplotlib.pyplot as plt
from data_aug import *
import cv2
# create image paths list
IMAGE_SIZE = 256
CH = 3
image_paths = []
#image_paths.append('sample/normal/img_5001.png')
image_paths.append('sample/normal/000001.png')
images = load_images(image_paths)
#plt.imshow(images[0])
#plt.show()
#quit()
# store returned list of numpy arrays
resize_imgs = tf_resize_images(images,IMAGE_SIZE,CH)
scaled_imgs = central_scale_images(resize_imgs, [0.90, 0.75, 0.60],IMAGE_SIZE,CH)
flipped_images = flip_images(resize_imgs,IMAGE_SIZE,CH)
translated_imgs = translate_images(resize_imgs,IMAGE_SIZE,CH)
rotated_imgs = rotate_images(resize_imgs,IMAGE_SIZE,CH)
salt_pepper_noise_imgs = add_salt_pepper_noise(resize_imgs,0.2,0.004)
gaussian_noise_imgs = add_gaussian_noise(resize_imgs)
# iterate through the array, and save the image to the directory
# data is array of numpy arrays
# so data[i] used to write, not data
for i in range(len(gaussian_noise_imgs)):
    #img = cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR)
    plt.imshow(gaussian_noise_imgs[i])
    #print(flipped_images[i].shape)
    plt.show()
    #cv2.imwrite("image_output" + str(i) + ".jpg", img)
    # unable to show the image output properly - to-do
# data = np.array(data)