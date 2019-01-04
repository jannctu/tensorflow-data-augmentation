import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.io import loadmat
from skimage.transform import resize


def load_ground_truth(gt_file_path,target_size):
    fuse_thick = []
    for gt_file in gt_file_path:
        gt = loadmat(gt_file)
        gt = gt['groundTruth'][0]

        #gt[9][0][0][2]
        # 9 =  fuse GT
        # 0 = fuse original
        # 1 =  fuse logical
        # 2 = fuse logical thicker
        bdry = gt[9][0][0][2]
        bdry = resize(bdry.astype(float), output_shape=target_size)
        fuse_thick.append(bdry)

    fuse_thick = np.concatenate([np.expand_dims(a, 0) for a in fuse_thick])
    fuse_thick = fuse_thick[..., np.newaxis]
    return fuse_thick