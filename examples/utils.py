# -*- coding: utf-8 -*-
# @Author: yanqiang
# @Date:   2018-05-10 22:31:37
# @Last Modified by:   yanqiang
# @Last Modified time: 2018-05-10 22:50:32
import os
import gzip
import shutil
import struct
import urllib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def huber_loss(labels, predictions, data=14.0):
    residual = tf.abs(labels - predictions)

    def f1(): return 0.5 * tf.square(residual)

    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < de, f1, f2)


def read_birth_life_data(filename):
    """
    Read in birth_life_2010.txt and return:
    data in the form of NumPy array
    n_samples: number of samples
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples
