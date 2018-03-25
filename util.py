#-*- coding:utf-8 -*-

#头文件
import numpy as np
import pandas as pd
import os
import tensorflow as tf

#读取数据
def load_data():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    os.chdir("data_9_feature")

    GM = pd.read_csv('GM.csv')
    Adware = pd.read_csv('adware.csv')
    Begin = pd.read_csv('begin.csv')

    GM = GM.sample(frac=1.0)
    Adware = Adware.sample(frac=1.0)
    Begin = Begin.sample(frac=1.0)

    train = pd.merge(Adware[:12000], GM[:4000], how='outer')
    train = pd.merge(train, Begin[:40000], how='outer')
    test = pd.merge(Adware[12000:15000], GM[4000:4575], how='outer')
    test = pd.merge(test, Begin[40000:50000], how='outer')

    train = train.sample(frac=1.0)
    test = test.sample(frac=1.0)

    train_X = train.loc[:,
              ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
               'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]
    train_Y = train.loc[:, ['calss']]
    test_X = test.loc[:,
             ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
              'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]
    test_Y = test.loc[:, ['calss']]

    train_X = train_X.values
    train_Y = train_Y.values
    test_X = test_X.values
    test_Y = test_Y.values

    train_Y=train_Y.reshape((-1,))
    train_Y = tf.one_hot(train_Y,depth=3,axis=1,dtype='float32')
    test_Y=test_Y.reshape((-1,))
    test_Y = tf.one_hot(test_Y,depth=3,axis=1,dtype='float32')

    with tf.Session() as sess:
        train_Y = sess.run(train_Y)
        test_Y = sess.run(test_Y)

    return train_X,train_Y,test_X,test_Y