# -*- coding:utf-8 -*-

import tensorflow as tf

INPUT_NODE = 81
OUTPUT_NODE = 3

# 定义函数，用于初始化权值 W，初始化偏置b
def weight_variable(shape, regularizer):
    weights=tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights

def bias_variable(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias

def inference(input_tensor, regularizer):

    layer=input_tensor

    with tf.variable_scope('BN1'):
        epsilon = 0.001
        mean, var = tf.nn.moments(layer, axes=[0], )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        layer = tf.nn.batch_normalization(layer, mean, var, offset=shift, scale=scale, variance_epsilon=epsilon)

    with tf.variable_scope('layer1'):
        input=81
        output=32
        weights = weight_variable([input, output], regularizer)
        biases = bias_variable([output])
        layer = tf.nn.relu(tf.matmul(layer, weights) + biases)
    '''
    with tf.variable_scope('BN2'):
        epsilon = 0.001
        mean, var = tf.nn.moments(layer, axes=[0], )
        scale = tf.Variable(tf.ones([32]))
        shift = tf.Variable(tf.zeros([32]))
        layer = tf.nn.batch_normalization(layer, mean, var, shift, scale, epsilon)
    '''
    with tf.variable_scope('layer2'):
        input=32
        output=64
        weights = weight_variable([input, output], regularizer)
        print(weights.name)
        biases = bias_variable([output])
        layer = tf.nn.relu(tf.matmul(layer, weights) + biases)
    '''
    with tf.variable_scope('BN3'):
        epsilon = 0.001
        mean, var = tf.nn.moments(layer, axes=[0], )
        scale = tf.Variable(tf.ones([64]))
        shift = tf.Variable(tf.zeros([64]))
        layer = tf.nn.batch_normalization(layer, mean, var, shift, scale, epsilon)
    '''
    with tf.variable_scope('layer3'):
        input=64
        output=32
        weights = weight_variable([input, output], regularizer)
        biases = bias_variable([output])
        layer = tf.nn.relu(tf.matmul(layer, weights) + biases)
    '''
    with tf.variable_scope('BN4'):
        epsilon = 0.001
        mean, var = tf.nn.moments(layer, axes=[0], )
        scale = tf.Variable(tf.ones([32]))
        shift = tf.Variable(tf.zeros([32]))
        layer = tf.nn.batch_normalization(layer, mean, var, shift, scale, epsilon)
    '''
    with tf.variable_scope('layer4'):
        input=32
        output=3
        weights = weight_variable([input, output], regularizer)
        biases = bias_variable([output])
        layer = tf.matmul(layer, weights) + biases

    return layer


