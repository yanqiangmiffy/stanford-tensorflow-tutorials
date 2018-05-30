# ================
# 贪婪执行 eager
# ================

import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
import utils

DATA_FILE = 'data/birth_life_2010.txt'

# 为了使用eager execution，必须在TensorFlow程序最开始的地方
# 调用tfe.enable_eager_execution()
tfe.enable_eager_execution()

# 加载数据
data, n_samples = utils.read_birth_life_data(DATA_FILE)
# print(data)
dataset = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))  # 创建切片数据集

# 创建变量Variables
w = tfe.Variable(0.0)
b = tfe.Variable(0.0)

# 定义linear predictor
def prediction(x):
    return x * w + b

# 定义 loss 函数 L(y,y_predicted)
def squared_loss(y, y_predicted):
    return (y - y_predicted)**2

def huber_loss(y, y_predicted, m=1.0):
    t = y - y_predicted
    return t**2 if tf.abs(t) < m else m * (2 * tf.abs(t) - m)

# train
def train(loss_fn):
    """使用损失函数loss_fn训练回归"""
    print("Training;loss function:" + loss_fn.__name__)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    def loss_for_example(x, y):
        return loss_fn(y, prediction(x))

    # `grad_fn(x_i, y_i)` returns (1) the value of `loss_for_example`
    # evaluated at `x_i`, `y_i` and (2) the gradients of any variables used in
    # calculating it.

    grad_fn = tfe.implicit_value_and_gradients(loss_for_example)

    start = time.time()
    for epoch in range(100):
        total_losss = 0.0
        for x_i, y_i in tfe.Iterator(dataset):
            loss, gradients = grad_fn(x_i, y_i)
            # 使用optimizer更新变量
            optimizer.apply_gradients(gradients)
            total_losss += loss
        if epoch % 10 == 0:
            print('Epoch{0}:{1}'.format(epoch, total_losss / n_samples))
    print('Took: %f seconds' % (time.time() - start))
    print('Eager execution exhibits significant overhead per operation. '
          'As you increase your batch size, the impact of the overhead will '
          'become less noticeable. Eager execution is under active development: '
          'expect performance to increase substantially in the near future!')
if __name__ == '__main__':
    train(huber_loss)
    plt.plot(data[:, 0], data[:, 1], 'bo')

    plt.plot(data[:, 0], data[:, 0] * w.numpy() + b.numpy(), 'r',
             label="huber regression")
    plt.legend()
    plt.show()
