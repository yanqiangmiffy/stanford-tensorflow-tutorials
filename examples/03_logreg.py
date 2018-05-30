# -*- coding: utf-8 -*-
# @Author: yanqaingmiffy
# @Date:   2018-05-20 10:37:07
# @Last Modified by:   yanqaingmiffy
# @Last Modified time: 2018-05-20 16:13:10

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import time
import utils

# 定义参数
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# 步骤1 ：读取数据
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# 步骤2 创建数据集和迭代器
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # 初始化训练时数据
test_init = iterator.make_initializer(test_data)  # 初始化训练集

# 创建weights和bias
# w初始化：平均值为0 方差为0.01
# b初始化：0
# b的shape 依赖于Y
w = tf.get_variable(name='weights', shape=(784, 10),
                    initializer=tf.random_normal_initializer(0, 0.001))
b = tf.get_variable(name='bias', shape=(
    1, 10), initializer=tf.zeros_initializer())

# 步骤 4：构建模型
# 模型将返回logits  logits: 未归一化的概率， 一般也就是 softmax的输入

logits = tf.matmul(img, w) + b

# 步骤5：定义损失函数 cross entropy
entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=label, name='entropy')
# 计算平均值
loss = tf.reduce_mean(entropy, name='loss')

# 步骤6:定义训练操作
# ensorFlow提供的tf.train.AdamOptimizer可控制学习速度。
# Adam 也是基于梯度下降的方法，但是每次迭代参数的学习步长都有一个确定的范围
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 步骤7：
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))  # 转换数据类型

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # 训练n 轮
    for i in range(n_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batchess = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batchess += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}:{1}'.format(i, total_loss / n_batchess))
    print("Total time:{0} seconds".format(time.time() - start_time))

    # 测试模型
    sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass
    print('Accuracy {0}'.format(total_correct_preds / n_test))
writer.close()
