# -*- coding:utf-8 -*-

# 头文件
import numpy as np
import pandas as pd
import os
import tensorflow as tf


def load_data(batch_size, is_training=True):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    os.chdir("/home/tonny-ftt/PycharmProjects/data")

    GM = pd.read_csv('GM.csv')
    Adware = pd.read_csv('Adware.csv')
    Begin = pd.read_csv('Begin.csv')

    GM = GM.sample(frac=1.0)
    Adware = Adware.sample(frac=1.0)
    Begin = Begin.sample(frac=1.0)

    train = pd.merge(Adware[:5000], GM[:4000], how='outer')
    train = pd.merge(train, Begin[:10000], how='outer')
    test = pd.merge(Adware[12000:15000], GM[4000:4575], how='outer')
    test = pd.merge(test, Begin[40000:50000], how='outer')
    train = train.sample(frac=1.0)
    test = test.sample(frac=1.0)

    train_X = train.iloc[:, :81]
    #train_X = train_X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    train_Y = train.iloc[:, 81]
    test_X = test.iloc[:, :81]
    #test_X = test_X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    test_Y = test.iloc[:, 81]

    train_X = train_X.values
    tr_Y = train_Y.values
    test_X = test_X.values
    test_Y = test_Y.values

    train_Y=tr_Y.reshape((-1,))
    train_Y = tf.one_hot(train_Y,depth=3,axis=1,dtype='float32')
    test_Y=test_Y.reshape((-1,))
    test_Y = tf.one_hot(test_Y,depth=3,axis=1,dtype='float32')

    with tf.Session() as sess:
        train_Y = sess.run(train_Y)
        test_Y = sess.run(test_Y)

    if is_training:
        tr_Y=tr_Y.astype('int32')
        num_tr_batch = 124000 // batch_size
        return train_X, train_Y, tr_Y,num_tr_batch

    else:

        num_te_batch = 30575 // batch_size
        return test_X, test_Y, num_te_batch



def get_batch_data(batch_size, num_threads):
    trX, trY,y, num_tr_batch = load_data(batch_size, is_training=True)
    trX=tf.cast(trX,tf.float32)
    trY=tf.cast(trY,tf.float32)
    data_queues = tf.train.slice_input_producer([trX, trY],shuffle=False)
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)
    return X, Y



def load_new_data():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    os.chdir("data_9_feature")

    data = pd.read_csv('Ambiguous_data.csv')
    num_features = data.shape[0]
    split = int(num_features * 0.8)
    train = data[:split]
    test = data[split:]

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
    tr_Y = train_Y.values
    test_X = test_X.values
    te_Y = test_Y.values

    train_Y = tr_Y.reshape((-1,))
    train_Y = tf.one_hot(train_Y, depth=3, axis=1, dtype='float32')
    test_Y = te_Y.reshape((-1,))
    test_Y = tf.one_hot(test_Y, depth=3, axis=1, dtype='float32')

    with tf.Session() as sess:
        train_Y = sess.run(train_Y)
        test_Y = sess.run(test_Y)

    return train_X, tr_y, train_Y, test_X, te_Y, test_Y



def main(argv=None):
    '''
    X,Y=get_batch_data(200,3)
    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess,coord=coord)
        print(X.eval(),Y.eval())
        coord.request_stop()
        coord.join(threads)
    fs=open('./data.txt','w')
    train_X, train_Y, tr_Y, num_tr = load_data(200, True)
    for step in range(num_tr):
        start=20*step
        end=start+20
        fs.write(str(step)+'==============='+str(train_X[start:end])+'=========='+str(train_Y[start:end]))
    fs.close()
    '''


if __name__ == '__main__':
    main()
