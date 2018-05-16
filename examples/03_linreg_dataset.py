#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-16 12:47:12
# @Author  : quincy
# @Email   ：yanqiang@cyou-inc.com
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = 'data/birth_life_2010.txt'

# 步骤1：读取数据
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# 步骤2：创建Dataset和迭代器
dataset = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))
iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

# 步骤3：创建weight和bias ，初始化为0

w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# 步骤4 创建模型
Y_predicted = X * w + b

# 步骤5 使用方差作为损失函数
loss = tf.square(Y - Y_predicted, name='loss')
# loss = utils.huber_loss(Y, Y_predicted)

# 步骤6 使用梯度下降算法 最小化loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()
with tf.Session() as sess:
    # 步骤7 初始化变量
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # 步骤 8：训练100 epochs
    for i in range(100):
        sess.run(iterator.initializer)
        total_loss = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
        except Exception as e:
            pass
        print('Epoch {0}:{1}'.format(i, total_loss / n_samples))
    # close writer
    writer.close()

    # 步骤9：
    w_out, b_out = sess.run([w, b])
    print('w:%f,b:%f' % (w_out, b_out))
print("Took:%f seconds" % (time.time() - start))

# 描绘结果
plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:, 0], data[:, 0] * w_out + b_out, 'r', label='Predicted data with squared error')
plt.legend()
plt.show()
