# -*- coding:utf-8 -*-

import tensorflow as tf
from util import load_data,get_batch_data
import network
import os
import pandas as pd
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

LEARNING_RATE_BASE = 100.0
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
EPOCH = 50
BATCH_SIZE=200

# 定义输入输出placeholder。
x = tf.placeholder(tf.float32, [None, network.INPUT_NODE], name='x-input')
y_ = tf.placeholder(tf.int32, [None,3], name='y-input')
# 建立模型
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
y = network.inference(x, regularizer)
# 定义损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
# 定义优化函数
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    1, LEARNING_RATE_DECAY,
    staircase=True)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_vars = optimizer.compute_gradients(loss)
train_step = optimizer.apply_gradients(grads_vars, global_step)
# 定义评估体系
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 读取数据
train_X, train_Y, tr_Y, num_tr=load_data(BATCH_SIZE,True)


def train():
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver(max_to_keep=1)
    # 训练模型
    ofs = open("./log.txt", 'w')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(EPOCH):
            for step in range(num_tr):
                start = step * BATCH_SIZE
                end = start + BATCH_SIZE
                _, loss_value, step,y_pre = sess.run([train_step, loss, global_step,y_], feed_dict={x: train_X[start:end], y_: train_Y[start:end]})
                for grad_val in grads_vars:
                    grad = sess.run(grad_val[0], feed_dict={x: train_X[start:end], y_: train_Y[start:end]})
                    # print(grad)
                ofs.write(str(grad_val[1])+"================= "+str(grad_val[0])+'\n')
                acc=sess.run(accuracy, feed_dict={x: train_X[start:end], y_: train_Y[start:end]})
                if step % 100 == 0:
                    print("After %d training step(s), loss on training batch is %g,accuracy is %a." % (step, loss_value,acc))
        ofs.close()
        saver.save(sess, 'net/my_net.ckpt')


def get_data():
    saver = tf.train.Saver()
    m, n = 0, 0
    result =np.arange(10)
    for i in range(train_X.shape[0]):
        X = train_X[i].reshape([1, 81])
        Y = train_Y[i].reshape([1, 3])
        with tf.Session() as sess:
            saver.restore(sess,'net/my_net.ckpt')
            y_pre = sess.run(y, feed_dict={x: X, y_: Y})
            y_pre = sess.run(tf.nn.softmax(y_pre))
            #print(y_pre)
            correct = sess.run(correct_prediction, feed_dict={x: X, y_: Y})
            if correct == 0:
                m = m + 1
            if max(y_pre[0]) <= 0.8:
                n = n + 1
            if max(y_pre[0]) <= 0.8 or correct == 0:
                res = np.c_[np.array(X),np.array([tr_Y[i]])]
                result=np.row_stack((result,res[0]))

    print(m, n, train_X.shape[0])
    np.savetxt('Ambiguous_data.csv',result,delimiter=',')


def main(argv=None):
    train()
    get_data()


if __name__ == '__main__':
    main()