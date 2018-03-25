#!/usr/bin/evn python3
# -*- coding:UTF-8 -*-

# 加载包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os

####################
#   CSV读取数据
# 只显示错误信息,不显示警告信息
# 数据集目录，数据集名称
# 数据集读取，训练集和测试集
####################

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.chdir("home/tonny-ftt/document/code/DNN")
TRAINING = "Iris-train.csv"
TEST = "Iris-test.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TEST,
    target_dtype=np.int,
    features_dtype=np.float32)


def input_fn_train():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y


def input_fn_eval():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y


####################
#   构建DNN结构
# 获取特征
# 构建DNN网络，3层，每层分别为100,200,100个节点
####################

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=9)]

classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[100, 200, 100],
    n_classes=4,
    activation_fn="relu",
    dropout=None,
    label_keys=None,
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001))

####################
#  训练和测试
# 拟合(训练)模型，迭代100*i步
# 测试模型，输出精度
####################

for i in range(200):
    classifier.fit(x=training_set.data, y=training_set.target, steps=100, batch_size=10000)
    acc = 0.0
    for j in range(6):
        accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target, batch_size=10000)["accuracy"]
        acc += accuracy_score
    acc = acc / 6
    print('Step: %d' % (i * 100), 'Accuracy: {0:f}'.format(acc))

'''
# batch_size与input_fn不可同时出现,inpit_fn与x,y等价
for i in range(200):
    classifier.fit(input_fn=input_fn_train(),steps=100)
    accuracy_score = classifier.evaluate(input_fn=input_fn_eval(),batch_size=10000)["accuracy"]
    print('Step: %d' % (i*100), 'Accuracy: {0:f}'.format(accuracy_score))
'''


