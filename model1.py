# -*- coding:utf-8 -*-

# 头文件
import numpy as np
import tensorflow as tf
from util import load_data, load_new_data
import pandas as pd

####################
#
#
#
#
####################
# 数据占位
x = tf.placeholder(tf.float32, [None, 9])
y_actual = tf.placeholder(tf.float32, shape=[None, 3])


# 定义函数，用于初始化权值 W，初始化偏置b
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 构建网络
with tf.variable_scope('input_layer'):
    h = tf.reshape(x, [-1, 9])
    W = weight_variable([9, 32])
    b = bias_variable([32])
    h = tf.nn.relu(tf.matmul(h, W) + b)

with tf.variable_scope('hiden_layer'):
    hiden = [128, 256, 64]
    input_node = 32
    for i in range(3):
        output_node = hiden[i]
        W = weight_variable([input_node, output_node])
        b = bias_variable([output_node])
        h = tf.nn.relu(tf.matmul(h, W) + b)
        input_node = output_node

with tf.variable_scope('output_layer'):
    keep_prob = tf.placeholder("float")
    h_drop = tf.nn.dropout(h, keep_prob)
    W = weight_variable([input_node, 3])
    b = bias_variable([3])
    y_predict = tf.nn.softmax(tf.matmul(h_drop, W) + b)

# 模型优化
'''
cross_entropy =tf.reduce_mean((y_actual-y_predict)*(y_actual-y_predict))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
'''
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_predict))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
init = tf.global_variables_initializer()

####################
#
#
#
#
####################

# 读取数据
train_X, tr_Y, test_X, te_Y = load_data()
train_Y = tr_Y.reshape((-1,))
train_Y = tf.one_hot(train_Y, depth=3, axis=1, dtype='float32')
test_Y = te_Y.reshape((-1,))
test_Y = tf.one_hot(test_Y, depth=3, axis=1, dtype='float32')

with tf.Session() as sess:
    train_Y = sess.run(train_Y)
    test_Y = sess.run(test_Y)

saver = tf.train.Saver()

# 训练与测试
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        sess.run(train_step, feed_dict={x: train_X, y_actual: train_Y, keep_prob: 0.3})
    saver.save(sess, 'net/my_net.ckpt')

# 数据过滤
with tf.Session() as sess:
    sess.run(init)
    m, n = 0, 0
    result = []
    for i in range(train_X.shape[0]):
        X = train_X[i].reshape([1, 9])
        Y = train_Y[i].reshape([1, 3])
        y_pre = sess.run(y_predict, feed_dict={x: X, y_actual: Y, keep_prob: 0.3})
        correct = sess.run(correct_prediction, feed_dict={x: X, y_actual: Y, keep_prob: 0.3})
        if correct == 0:
            m = m + 1
        if max(max(y_pre)) <= 0.8:
            n = n + 1
        if max(max(y_pre)) <= 0.8 or correct == 0:
            res = train_X[i].tolist()
            res.extend(tr_Y[i])
            result.append(res)
    print(m, n, train_X.shape[0])

pd_data = pd.DataFrame(np.array(result),
                       columns=['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin',
                                'bVarianceDataBytes',
                                'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward', 'calss'])
pd_data.to_csv('Ambiguous_data.csv')
