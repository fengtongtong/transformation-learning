#-*- coding:utf-8 -*-

#头文件
import numpy as np
import tensorflow as tf
from util import load_data


#数据占位
x = tf.placeholder(tf.float32,[None,9])
y_actual = tf.placeholder(tf.float32, shape=[None,3])


#定义函数，用于初始化权值 W，初始化偏置b
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#构建网络
with tf.variable_scope('input_layer'):
	h = tf.reshape(x,[-1,9])
	W = weight_variable([9,32])
	b = bias_variable([32])
	h = tf.nn.relu(tf.matmul(h, W) + b)

with tf.variable_scope('hiden_layer'):
	hiden[3]=[128,256,64]
	input_node=32
	for i in range(3):
		output_node=hiden[i]
		W = weight_variable([input_node,output_node])
		b = bias_variable([output_node])
		h = tf.nn.relu(tf.matmul(h, W) + b)
		input_node=output_node

with tf.variable_scope('output_layer'):
	keep_prob = tf.placeholder("float")
	h_drop = tf.nn.dropout(h, keep_prob)
	W = weight_variable([input_node,3])
	b = bias_variable([3])
	y_predict=tf.nn.softmax(tf.matmul(h_drop, W) + b)



#模型优化
cross_entropy =tf.reduce_mean((y_actual-y_predict)*(y_actual-y_predict))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


#读取数据
train_X,train_Y,test_X,test_Y=load_data()


#训练与测试
for i in range(1000):
	train_step.run(feed_dict={x: train_X, y_actual: sess.run(train_Y), keep_prob: 0.5})
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: train_X, y_actual:train_Y,keep_prob: 0.5})
		print("step %d, training accuracy %g" % (i, train_accuracy))
		test_accuracy = accuracy.eval(feed_dict={x:test_X, y_actual:test_Y, keep_prob: 0.5})
		print("step %d, testing accuracy %g" % (i, test_accuracy))
	#print(sess.run(y_predict,feed_dict={x: train_X, y_actual: sess.run(train_Y), keep_prob: 0.5}))
