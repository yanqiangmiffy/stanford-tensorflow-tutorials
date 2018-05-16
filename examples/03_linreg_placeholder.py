import os
os.environ['TF_CCP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = 'data/birth_life_2010.txt'

# 步骤 1：读取数据
data, n_samples = utils.read_birth_life_data(DATA_FILE)
# print(data, n_samples)

# 步骤 2:给X和Y创建占位符
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 步骤 3：创建weights 和 bias ,初始化为0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# 步骤 4：创建模型
Y_predicted = w * X + b

# 步骤 5:使用方差squared error 作为损失函数 loss function
# 也可以使用其他平均方差作为损失函数 或者 Huber loss
loss = tf.square(Y - Y_predicted, name='loss')
# loss = utils.huber_loss(Y, Y_predicted)

# 步骤 6：使用梯度下降算法最小化损失， 学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.001).minimize(loss)

start = time.time()
writer = tf.summary.FileWriter('./graphs/linear_reg', tf.get_default_graph())
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 训练模型 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print('Epoch {0}:{1}'.format(i, total_loss / n_samples))

    # 关闭witer
    writer.close()

    # 步骤 9：输出 w 和 b
    w_out, b_out = sess.run([w, b])
print('Took :%f seconds' % (time.time() - start))

# 画图
plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:, 0], data[:, 0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
