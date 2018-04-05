# -*- coding:utf-8 -*-

import tensorflow as tf

INPUT_NODE = 9
OUTPUT_NODE = 3
LAYER1_NODE,LAYER2_NODE,LAYER3_NODE,LAYER4_NODE=[128,64,32,16]

# 定义函数，用于初始化权值 W，初始化偏置b
def weight_variable(shape, regularizer):
    weights=tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights

def bias_variable(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias

def inference(input_tensor, regularizer, model):

    layer = input_tensor
    name_list = []

    with tf.variable_scope('BN1'):
        epsilon = 0.001
        mean, var = tf.nn.moments(layer, axes=[0], )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        name_list.append(scale)
        name_list.append(shift)
        layer = tf.nn.batch_normalization(layer, mean, var, offset=shift, scale=scale, variance_epsilon=epsilon)

    with tf.variable_scope('layer1'):
        input = INPUT_NODE
        output = LAYER1_NODE
        weights = weight_variable([input, output], regularizer)
        biases = bias_variable([output])
        name_list.append(weights)
        name_list.append(biases)
        layer = tf.nn.relu(tf.matmul(layer, weights) + biases)

    with tf.variable_scope('BN2'):
        epsilon = 0.001
        mean, var = tf.nn.moments(layer, axes=[0], )
        scale = tf.Variable(tf.ones([LAYER1_NODE]))
        shift = tf.Variable(tf.zeros([LAYER1_NODE]))
        name_list.append(scale)
        name_list.append(shift)
        layer = tf.nn.batch_normalization(layer, mean, var, offset=shift, scale=scale, variance_epsilon=epsilon)

    with tf.variable_scope('layer2'):
        input=LAYER1_NODE
        output=LAYER2_NODE
        weights = weight_variable([input, output], regularizer)
        biases = bias_variable([output])
        name_list.append(weights)
        name_list.append(biases)
        layer = tf.nn.relu(tf.matmul(layer, weights) + biases)
    '''
    with tf.variable_scope('BN3'):
        epsilon = 0.001
        mean, var = tf.nn.moments(layer, axes=[0], )
        scale = tf.Variable(tf.ones([LAYER2_NODE]))
        shift = tf.Variable(tf.zeros([LAYER2_NODE]))
        name_list.append(scale)
        name_list.append(shift)
        layer = tf.nn.batch_normalization(layer, mean, var, shift, scale, epsilon)
    '''
    with tf.variable_scope('layer3'):
        input=LAYER2_NODE
        output=LAYER3_NODE
        weights = weight_variable([input, output], regularizer)
        biases = bias_variable([output])
        name_list.append(weights)
        name_list.append(biases)
        layer3 = tf.nn.relu(tf.matmul(layer, weights) + biases)
    '''
    with tf.variable_scope('BN4'):
        epsilon = 0.001
        mean, var = tf.nn.moments(layer, axes=[0], )
        scale = tf.Variable(tf.ones([LAYER3_NODE]))
        shift = tf.Variable(tf.zeros([LAYER3_NODE]))
        name_list.append(scale)
        name_list.append(shift)
        layer = tf.nn.batch_normalization(layer, mean, var, shift, scale, epsilon)
    '''
    if model==1:
        with tf.variable_scope('layer4'):
            input=LAYER3_NODE
            output=OUTPUT_NODE
            weights = weight_variable([input, output], regularizer)
            biases = bias_variable([output])
            name_list.append(weights)
            name_list.append(biases)
            layer = tf.matmul(layer3, weights) + biases
            layer_pre=tf.nn.softmax(layer)

        return layer,layer_pre,name_list

    elif model==2:
        with tf.variable_scope('layer5'):
            input=LAYER3_NODE
            output=OUTPUT_NODE
            weights = weight_variable([input, output], regularizer)
            biases = bias_variable([output])
            name_list.append(weights)
            name_list.append(biases)
            layer = tf.matmul(layer3, weights) + biases
            layer_pre=tf.nn.softmax(layer)

        return layer, layer_pre, name_list

    elif model==3:
        with tf.variable_scope('layer4'):
            input=LAYER3_NODE
            output=OUTPUT_NODE
            weights = weight_variable([input, output], regularizer)
            biases = bias_variable([output])
            name_list.append(weights)
            name_list.append(biases)
            layer4 = tf.matmul(layer3, weights) + biases

        with tf.variable_scope('layer5'):
            input = LAYER3_NODE
            output = OUTPUT_NODE
            weights = weight_variable([input, output], regularizer)
            biases = bias_variable([output])
            name_list.append(weights)
            name_list.append(biases)
            layer5 = tf.matmul(layer3, weights) + biases

        layer_new = tf.concat([layer4, layer5], 1)

        with tf.variable_scope('layer6'):
            input = OUTPUT_NODE*2
            output = OUTPUT_NODE
            weights = weight_variable([input, output], regularizer)
            biases = bias_variable([output])
            name_list.append(weights)
            name_list.append(biases)
            layer = tf.matmul(layer_new, weights) + biases
            layer_pre = tf.nn.softmax(layer)

        return layer,layer_pre,name_list

    else:
        print('Wrong!')

