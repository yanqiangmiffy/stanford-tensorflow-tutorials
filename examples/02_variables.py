import os
os.environ['TF_CCP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# Example 1: creating variables
s = tf.Variable(2, name='scalar')
m = tf.Variable([[0, 1], [2, 3]], name='matrix')
W = tf.Variable(tf.zeros([784, 10]), name='big_matrix')
# tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
# 这个函数产生正太分布，均值和标准差自己设定。
V = tf.Variable(tf.truncated_normal([784, 10]), name='normal_matrix')


s = tf.get_variable('scalar', initializer=tf.constant(2))
m = tf.get_variable('matrix', initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable('big_matrix', shape=(784, 10),
                    initializer=tf.zeros_initializer())
V = tf.get_variable('normal_matrix', shape=(784, 10),
                    initializer=tf.truncated_normal_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(V.eval())

# Example 2:assigning values to variables
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W))  # >> 10

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(assign_op)
    print(W.eval())  # >> 100

# create a variable whose orginal value is 2
# https://blog.csdn.net/u012436149/article/details/53696970
# 使用tf.Variable时，如果检测到命名冲突，系统会自己处理。使用tf.get_variable()时，系统不会处理冲突，而会报错
with tf.variable_scope('scalar', reuse=False):
    a = tf.get_variable('scalar', initializer=tf.constant(2))
    a_times_two = a.assign(a * 2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a_times_two))					# >> 4
        print(sess.run(a_times_two))                 	# >> 8
        print(sess.run(a_times_two))                	# >> 16

W = tf.Variable(10)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W.assign_add(10)))                   # >> 20
    print(sess.run(W.assign_sub(2)))                    # >> 18

# Example 3: Each session has its own copy of variable 每个session 变量独立，互不影响
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10)))                      # >> 20
print(sess2.run(W.assign_sub(2)))                       # >> 8
print(sess1.run(W.assign_add(100)))                     # >> 120
print(sess2.run(W.assign_sub(50)))                      # >> -42
sess1.close()
sess2.close()

# Example 4: create a variable with the initial value depending on another
# variable
W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(W * 2)
