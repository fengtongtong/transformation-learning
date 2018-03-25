# -*- coding:utf-8 -*-

# 头文件
import numpy as np
import pandas as pd
import os
import tensorflow as tf


# 读取数据
def load_data():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    os.chdir("data_9_feature")
    GM, Adware, Begin = pd.read_csv('GM.csv'), pd.read_csv('adware.csv'), pd.read_csv('begin.csv')
    GM, Adware, Begin = GM.sample(frac=1.0),Adware.sample(frac=1.0), Begin.sample(frac=1.0)
    train = pd.merge(pd.merge(Adware[:12000], GM[:3600], how='outer'), Begin[:40000], how='outer')
    test = pd.merge(pd.merge(Adware[12000:15000], GM[3600:4500], how='outer'), Begin[40000:50000], how='outer')
    train, test = train.sample(frac=1.0), test.sample(frac=1.0)
    train_X = (train.loc[:,
               ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
                'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]).values
    train_Y = (train.loc[:, ['calss']]).values
    test_X = (test.loc[:,
              ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
               'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]).values
    test_Y = (test.loc[:, ['calss']]).values
    train_Y = train_Y.reshape((-1,))
    train_Y = tf.one_hot(train_Y, depth=3, axis=1, dtype='float32')
    test_Y = test_Y.reshape((-1,))
    test_Y = tf.one_hot(test_Y, depth=3, axis=1, dtype='float32')

    with tf.Session() as sess:
        train_Y = sess.run(train_Y)
        test_Y = sess.run(test_Y)

    return train_X, train_Y, test_X, test_Y