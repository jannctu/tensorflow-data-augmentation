import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np
import sys
import matplotlib.pyplot as plt

def load_images(X_img_file_paths):
    images = []
    for index, file_path in enumerate(X_img_file_paths):
        im = mpimg.imread(file_path)  # Do not read alpha channel.
        #im = imread(file_path)
        #print(len(im.shape))
        #quit()
        if len(im.shape) < 3:
            images.append(np.expand_dims(np.expand_dims(im, 2), 0))
        else:
            images.append(np.expand_dims(im, 0))

    images = np.concatenate(images)
    #sz = images.shape
    #if len(sz) < 4:
    #X_data = np.reshape(images, (sz[0],sz[1],sz[2],1))
    #else:

    return images

def tf_resize_images(X_imgs,IMAGE_SIZE,CH):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, CH))
    tf_img = tf.image.resize_images(X, [IMAGE_SIZE, IMAGE_SIZE], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            resized_img = sess.run(tf_img, feed_dict={X: img})
            X_data.append(resized_img)
    X_data = np.array(X_data, dtype=np.float32)  # Convert to numpy
    return X_data

def central_scale_images(X_imgs, scales,IMAGE_SIZE,CH):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype=np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype=np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, CH))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis=0)
            scaled_imgs = sess.run(tf_img, feed_dict={X: batch_img})
            X_scale_data.extend(scaled_imgs)

    X_scale_data = np.array(X_scale_data, dtype=np.float32)
    return X_scale_data

def flip_images(X_imgs,IMAGE_SIZE,CH):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, CH))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict={X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype=np.float32)
    return X_flip


from math import ceil, floor

def get_translate_parameters(index,IMAGE_SIZE):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE))
    else:  # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE

    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs,IMAGE_SIZE,CH):
    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    n_translations = 4
    X_translated_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, CH),
                                    dtype=np.float32)
            X_translated.fill(1.0)  # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i,IMAGE_SIZE)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
            w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype=np.float32)
    return X_translated_arr

def rotate_images(X_imgs, IMAGE_SIZE,CH):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, CH))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k=k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict={X: img, k: i + 1})
                X_rotate.append(rotated_img)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate


def add_salt_pepper_noise(X_imgs,salt_vs_pepper,amount):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy


import cv2

def add_gaussian_noise(X_imgs,p1,p2):
    gaussian_noise_imgs = []
    row, col, ch = X_imgs[0].shape

    #sigma = var ** 0.5
    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        #gaussian = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
        if ch > 1:
            gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        #gaussian_img = cv2.addWeighted(X_img, 0.9, 0.1 * gaussian, 0.1, 0)
        gaussian_img = cv2.addWeighted(X_img, p1, p2 * gaussian, p2, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs