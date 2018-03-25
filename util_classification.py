import os
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.chdir("data_9_feature")


def load_GM_data(batch_size, is_trainning):
    GM = pd.read_csv('GM.csv')
    GM = GM.sample(frac=1.0)

    if is_trainning == True:
        train_X = GM.loc[:,
                  ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
                   'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]
        train_Y = GM.loc[:, ['calss']]

        return trainX, trainY

    else:
        test_X = GM.loc[:,
                 ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
                  'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]
        test_Y = GM.loc[:, ['calss']]


def load_Adware_data(batch_size):
    Adware = pd.read_csv('adware.csv')
    Adware = Adware.sample(frac=1.0)

    train_X = Adware.loc[:,
              ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
               'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]
    train_Y = Adware.loc[:, ['calss']]
    test_X = Adware.loc[:,
             ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
              'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]
    test_Y = Adware.loc[:, ['calss']]

    train_X = train_X.values
    train_Y = train_Y.values
    test_X = test_X.values
    test_Y = test_Y.values

    train_Y = train_Y.reshape((-1,))
    train_Y = tf.one_hot(train_Y, depth=3, axis=1, dtype='float32')
    test_Y = test_Y.reshape((-1,))
    test_Y = tf.one_hot(test_Y, depth=3, axis=1, dtype='float32')

    with tf.session() as sess:
        train_Y = sess.run(train_Y)
        test_Y = sess.run(test_Y)

    num_tr_batch = 155613 // batch_size

    return trainX, trainY, num_tr_batch


def load_Begin_data(batch_size):
    Begin = pd.read_csv('begin.csv')
    Begin = Begin.sample(frac=1.0)

    train_X = Begin.loc[:,
              ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
               'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]
    train_Y = Begin.loc[:, ['calss']]
    test_X = Begin.loc[:,
             ['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin', 'bVarianceDataBytes',
              'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward']]
    test_Y = Begin.loc[:, ['calss']]

    train_X = train_X.values
    train_Y = train_Y.values
    test_X = test_X.values
    test_Y = test_Y.values

    train_Y = train_Y.reshape((-1,))
    train_Y = tf.one_hot(train_Y, depth=3, axis=1, dtype='float32')
    test_Y = test_Y.reshape((-1,))
    test_Y = tf.one_hot(test_Y, depth=3, axis=1, dtype='float32')

    with tf.session() as sess:
        train_Y = sess.run(train_Y)
        test_Y = sess.run(test_Y)

    num_tr_batch = 471597 // batch_size

    return trainX, trainY, num_tr_batch


def get_GM_data(batch_size):
    trX, trY, num_tr_batch = load_GM_data()
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=8,
                                  batch_size=batch_size,
                                  capacity=batch_size * 4,
                                  min_after_dequeue=batch_size * 2,
                                  allow_smaller_final_batch=False)

    return (X, Y)


def get_Adware_data(batch_size):
    trX, trY, num_tr_batch = load_Adware_data()
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=8,
                                  batch_size=batch_size,
                                  capacity=batch_size * 4,
                                  min_after_dequeue=batch_size * 2,
                                  allow_smaller_final_batch=False)

    return (X, Y)


def get_Begin_data(batch_size):
    trX, trY, num_tr_batch = load_Begin_data()
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=8,
                                  batch_size=batch_size,
                                  capacity=batch_size * 4,
                                  min_after_dequeue=batch_size * 2,
                                  allow_smaller_final_batch=False)

    return (X, Y)


def get_batch_data():
    GX, GY, Gnum = get_GM_data(4000)
    AX, AY, Anum = get_Adware_data(12000)
    BX, BY, Bnum = get_Begin_data(40000)



def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return (scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs
