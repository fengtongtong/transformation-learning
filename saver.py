# -*- coding:utf-8 -*-

# 头文件
import numpy as np
import tensorflow as tf
from util import load_data
import pandas as pd


#构建模型
learning_rate = 0.001
model_path = "/tmp/model.ckpt"
hiden = [32,128, 256, 64]
n_feature = 9 # MNIST data input (img shape: 28*28)
n_classes = 3 # MNIST total classes (0-9 digits)

x = tf.placeholder(tf.float32, [None,n_feature])
y_actual = tf.placeholder(tf.float32, shape=[None, n_classes])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

with tf.variable_scope('input_layer'):
    output_node = hiden[0]
    h = tf.reshape(x, [-1, n_feature])
    W = weight_variable([n_feature, output_node])
    b = bias_variable([output_node])
    h = tf.nn.relu(tf.matmul(h, W) + b)

with tf.variable_scope('hiden_layer'):
    for i in range(1,4):
        input_node = hiden[i-1]
        output_node = hiden[i]
        W = weight_variable([input_node, output_node])
        b = bias_variable([output_node])
        h = tf.nn.relu(tf.matmul(h, W) + b)

with tf.variable_scope('output_layer'):
    input_node=hiden[-1]
    keep_prob = tf.placeholder("float")
    h_drop = tf.nn.dropout(h, keep_prob)
    W = weight_variable([input_node, n_classes])
    b = bias_variable([n_classes])
    y_predict = tf.nn.softmax(tf.matmul(h_drop, W) + b)

# 读取数据
train_X, tr_Y, test_X, te_Y = load_data()
train_Y = tr_Y.reshape((-1,))
train_Y = tf.one_hot(train_Y, depth=3, axis=1, dtype='float32')
test_Y = te_Y.reshape((-1,))
test_Y = tf.one_hot(test_Y, depth=3, axis=1, dtype='float32')

with tf.Session() as sess:
    train_Y = sess.run(train_Y)
    test_Y = sess.run(test_Y)

# 模型优化
'''
cross_entropy =tf.reduce_mean((y_actual-y_predict)*(y_actual-y_predict))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
'''
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_predict))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 训练与测试
init = tf.global_variables_initializer()
saver = tf.train.Saver()   #括号中也可以添加参数列表，进行限定需要保存的参数
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        sess.run(train_step, feed_dict={x: train_X, y_actual: train_Y, keep_prob: 0.3})
    saver.save(sess, 'net/my_net.ckpt')
#获取模型参数变量名
    all_vars=tf.trainable_variables()
    for v in all_vars:
        print(v.name)




#加载模型
import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.chdir("data_9_feature")

# Let's laod a previous meta graph in the current graph in use: usually the default graph
# This actions returns a Saver
# 恢复操作和元数据
# 将恢复的图载入到当前的默认图中。所有当你完成模型恢复之后，你可以在默认图中访问载入的任何内容，比如一个张量，一个操作或者集合。
saver = tf.train.import_meta_graph('net/my_net.ckpt.meta')
# We can now access the default graph where all our metadata has been loaded
graph = tf.get_default_graph()
#获取模型中所有的参数
x1 = graph.get_tensor_by_name('input_layer/Variable:0')
x2 = graph.get_tensor_by_name('input_layer/Variable_1:0')
x3 = graph.get_tensor_by_name('hiden_layer/Variable:0')
x4 = graph.get_tensor_by_name('hiden_layer/Variable_1:0')
x5 = graph.get_tensor_by_name('hiden_layer/Variable_2:0')
x6 = graph.get_tensor_by_name('hiden_layer/Variable_3:0')
x7 = graph.get_tensor_by_name('hiden_layer/Variable_4:0')
x8 = graph.get_tensor_by_name('hiden_layer/Variable_5:0')
x9 = graph.get_tensor_by_name('output_layer/Variable:0')
x10 = graph.get_tensor_by_name('output_layer/Variable_1:0')

# 恢复权重
saver = tf.train.Saver()
with tf.Session() as sess:
    # To initialize values with saved data
    saver.restore(sess, 'net/my_net.ckpt')
    # print(sess.run(x1)) # returns 1000

# 多个图的连接，一个图的输出作为另一个图的输入
output_conv=graph.get_tensor_by_name('hiden_layer/Variable_5:0')
# Stop the gradient for fine-tuning
output_conv_sg = tf.stop_gradient(x9)
# It's an identity function
output_conv_shape = output_conv_sg.get_shape().as_list()
print(output_conv_shape)
# Build further operations
with tf.variable_scope('FC_layer'):
    #keep_prob = tf.placeholder("float")
    #h_drop = tf.nn.dropout(output_conv_sg, keep_prob)
    W = tf.get_variable('W',shape=[output_conv_shape[1], 3], initializer=tf.random_normal_initializer(stddev=1e-1))
    b = tf.get_variable('b',shape=[3],initializer=tf.constant_initializer(0.1))
    y_predict = tf.nn.softmax(tf.matmul(output_conv_sg, W) + b)

