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
train2_X, tr2_Y, test2_X, te2_Y = load_new_data()
train2_Y = tr2_Y.reshape((-1,))
train2_Y = tf.one_hot(train2_Y, depth=3, axis=1, dtype='float32')
test2_Y = te2_Y.reshape((-1,))
test2_Y = tf.one_hot(test2_Y, depth=3, axis=1, dtype='float32')

with tf.Session() as sess:
    train2_Y = sess.run(train2_Y)
    test2_Y = sess.run(test2_Y)

saver = tf.train.Saver()

# 训练与测试
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10000):
        sess.run(train_step, feed_dict={x: train2_X, y_actual: train2_Y, keep_prob: 0.3})
        if epoch % 400 == 0:
            test_accuracy = accuracy.eval(feed_dict={x: test2_X, y_actual: test2_Y, keep_prob: 0.3})
            print("step %d, testing accuracy %g" % (epoch, test_accuracy))
    saver.save(sess, 'net2/my_net.ckpt')

