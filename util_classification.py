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

    GM = pd.read_csv('GM.csv')
    Adware = pd.read_csv('adware.csv')
    Begin = pd.read_csv('begin.csv')

    GM = GM.sample(frac=1.0)
    Adware = Adware.sample(frac=1.0)
    Begin = Begin.sample(frac=1.0)

    train = pd.merge(Adware[:5000], GM[:4000], how='outer')
    train = pd.merge(train, Begin[:10000], how='outer')
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
    tr_Y = train_Y.values
    test_X = test_X.values
    te_Y = test_Y.values

    train_Y=tr_Y.reshape((-1,))
    train_Y = tf.one_hot(train_Y,depth=3,axis=1,dtype='float32')
    test_Y=te_Y.reshape((-1,))
    test_Y = tf.one_hot(test_Y,depth=3,axis=1,dtype='float32')

    with tf.Session() as sess:
        train_Y = sess.run(train_Y)
        test_Y = sess.run(test_Y)

    return train_X, tr_Y,train_Y, test_X, te_Y,test_Y


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


# -*- coding:utf-8 -*-

import tensorflow as tf
from util import load_data
import network
import os
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
#os.chdir("data_9_feature")

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "net/"
MODEL_NAME = "model"

# 定义输入输出placeholder。
x = tf.placeholder(tf.float32, [None, network.INPUT_NODE], name='x-input')
y_ = tf.placeholder(tf.float32, [None, network.OUTPUT_NODE], name='y-input')
# 建立模型
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
y = network.inference(x, regularizer)
# 定义滑动平均操作。
global_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
# 定义损失函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
# 定义优化函数
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    1, LEARNING_RATE_DECAY,
    staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')
# 定义评估体系
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 读取数据
train_X, tr_Y, train_Y= load_data()
#train_X, tr_Y, train_Y, test_X, te_Y, test_Y = load_data()


def train():
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver(max_to_keep=1)
    # 训练模型
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        max_acc=0
        for i in range(TRAINING_STEPS):
            _, loss_value, step= sess.run([train_op, loss, global_step], feed_dict={x: train_X, y_: train_Y})
            acc=sess.run(accuracy, feed_dict={x: train_X, y_: train_Y})
            if i % 1 == 0:
                print("After %d training step(s), loss on training batch is %g,accuracy is %a." % (step, loss_value,acc))
                if acc>max_acc:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def get_data():
    saver = tf.train.Saver()
    m, n = 0, 0
    result = []
    for i in range(train_X.shape[0]):
        X = train_X[i].reshape([1, 9])
        Y = train_Y[i].reshape([1, 3])
        with tf.Session() as sess:
            saver.restore(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
            y_pre = sess.run(y, feed_dict={x: X, y_: Y})
            y_pre = tf.nn.softmax(y_pre)
            correct = sess.run(correct_prediction, feed_dict={x: X, y_: Y})
            if correct == 0:
                m = m + 1
            if max(max(y_pre)) <= 0.8:
                n = n + 1
            if max(max(y_pre)) <= 0.8 or correct == 0:
                res = train_X[i].tolist()
                res.extend(tr_Y[i])
                result.append(res)
        print(m, n, train_X.shape[0])


    pd_data = pd.DataFrame(np.array(result),
                           columns=['total_fpktl', 'total_bpktl', 'min_flowpktl', 'max_flowpktl', 'flow_fin',
                                    'bVarianceDataBytes',
                                    'max_idle', 'Init_Win_bytes_forward', 'min_seg_size_forward', 'calss'])
    pd_data.to_csv('Ambiguous_data.csv')


def main(argv=None):
    train()
    get_data()


if __name__ == '__main__':
    main()