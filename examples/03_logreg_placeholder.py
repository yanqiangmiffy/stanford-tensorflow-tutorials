#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-05 17:00:43
# @Author  : quincy
# @Email   ：yanqiang@cyou-inc.com

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

import utils

# 定义模型的参数
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# 步骤1：读取数据
# batch_size：每批数据量的大小。
# DL通常用SGD的优化算法进行训练，也就是一次（1 个iteration）一起训练batchsize个样本，计算它们的平均损失函数值，来更新参数。

mnist = input_data.read_data_sets('data/mnist', one_hot=True)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 这个使用tensorflow框架下载，效果一样
X_batch, Y_batch = mnist.train.next_batch(batch_size)


# 步骤2：为特征和标签创建占位符placeholder
# 每张图片是28*28=784，因此一张图片用1x784的tensor标示
# 每张图片属于10类之一，0-9十个数字,每个标签是一个one hot 向量,比如图片“1”，[0,1,0,0,0,0,0,0,0,0]
X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='image')

# 步骤3：创建权重和偏置
# w为随机变量，服从平均值为0，标准方差(stddev)为0.01的正态分布
# b初始化为0
# w的shape取决于X和Y  Y = tf.matmul(X, w)
# b的shape取决于Y
# Y=Xw+b  [1,10]=[1,784][784,10]+[1,10]
w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# 步骤4：创建模型
logits = tf.matmul(X, w) + b

# 步骤5:定义损失函数
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)  # 计算一个batch下的loss平均值
# loss = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(logits) * tf.log(Y), reduction_indices=[1]))

# 步骤6：定义训练optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 步骤7：计算测试集的准确度
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg_placeholder', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples / batch_size)
    print(n_batches)

    # 训练模型 n_epochs 次
    for i in range(n_epochs):
        total_loss = 0
        for j in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], {X: X_batch, Y: Y_batch})
            total_loss += loss_batch
        print("Average loss opoch {0}:{1}".format(i, total_loss / n_batches))
    print("Total time:{0}seconds".format(time.time() - start_time))

    # 测试模型

    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run(accuracy, {X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch
    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))
writer.close()
