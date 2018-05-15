# -*- coding: utf-8 -*-
# @Author: yanqiang
# @Date:   2018-05-13 10:37:40
# @Last Modified by:   yanqiang
# @Last Modified time: 2018-05-13 11:41:55

import os


# 在tensorflow的log日志等级如下：
# - 0：显示所有日志（默认等级）
# - 1：显示info、warning和error日志
# - 2：显示warning和error信息
# - 3：显示error日志信息
# 保持默认日志等级时候，tensorflow执行会出现类似以下警告：
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

# 例子1：简单创建 log writer
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
writer = tf.summary.FileWriter('./graphs/simple', tf.get_default_graph())
with tf.Session() as sess:
    # writer=tf.summary.FileWriter('./graphs',sess.graph)
    print(sess.run(x))
writer.close()

# 例子2：div的奇思妙用
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
with tf.Session() as sess:
    print(sess.run(tf.div(b, a)))  # 对应元素相除， 取商数
    print(sess.run(tf.divide(b, a)))  # 对应元素相除
    print(sess.run(tf.truediv(b, a)))  # 对应元素 相除
    # print(sess.run(tf.realdiv(b, a)))
    print(sess.run(tf.floordiv(b, a)))  # 结果向下取整, 但结果dtype与输入保持一致
    print(sess.run(tf.truncatediv(b, a)))  # 对应元素 截断除 取余
    print(sess.run(tf.floor_div(b, a)))

# 例子3：乘法
a = tf.constant([10, 20], name='a')
b = tf.constant([2, 3], name='b')
with tf.Session() as sess:
    print(sess.run(tf.multiply(a, b)))
    print(sess.run(tf.tensordot(a, b, 1)))

# 例子4：Python 基础数据类型
t_0 = 19
x = tf.zeros_like(t_0)
y = tf.ones_like(t_0)
print(x)
print(y)

t_1 = ['apple', 'peach', 'banana']
x = tf.zeros_like(t_1) 					# ==> ['' '' '']
# y = tf.ones_like(t_1)                           # ==> TypeError:

t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]
x = tf.zeros_like(t_2) 					# ==> 3x3 tensor, all elements are False
y = tf.ones_like(t_2) 					# ==> 3x3 tensor, all elements are True

print(tf.int32.as_numpy_dtype())

# Example 5: printing your graph's definition
my_const = tf.constant([1.0, 2.0], name='my_const')
print(tf.get_default_graph().as_graph_def())
