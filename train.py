# -*- coding:utf-8 -*-

import tensorflow as tf
from util import load_batch
import network
import os
import numpy as np



os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
#os.chdir("/home/tonny-ftt/PycharmProjects/data_9_feature")
file_name_string = 'data_new.csv'
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
EPOCH_1 = 3000
EPOCH_2 = 56000
BATCH_SIZE_1 = 500
BATCH_SIZE_2 = 1



# 定义输入输出placeholder。
x = tf.placeholder(tf.float32, [None, network.INPUT_NODE], name='x-input')
y_ = tf.placeholder(tf.int32, [None,], name='y-input')

# 建立模型
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
y = network.inference(x, regularizer)

# 定义损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

# 定义优化函数
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    1, LEARNING_RATE_DECAY,
    staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# grads_vars = optimizer.compute_gradients(loss)
# train_step = optimizer.apply_gradients(grads_vars, global_step)

# 定义评估体系
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1),dtype=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 读取数据
#train_X, train_Y, tr_Y, num_tr=load_data(BATCH_SIZE,True)
example_batch_1, label_batch_1=load_batch([file_name_string],BATCH_SIZE_1)
init=tf.global_variables_initializer()


def train():
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver(max_to_keep=1)
    # 训练模型
    for var in tf.trainable_variables():
        print(var.name)
    w1 = tf.get_default_graph().get_tensor_by_name('layer1/Variable:0')
    w2 = tf.get_default_graph().get_tensor_by_name('layer2/Variable:0')
    w3 = tf.get_default_graph().get_tensor_by_name('layer3/Variable:0')
    w4 = tf.get_default_graph().get_tensor_by_name('layer4/Variable:0')
    ofs = open("./log.txt", 'w')
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for i in range(EPOCH_1):
                example, label = sess.run([example_batch_1, label_batch_1])
                # print([example,label])
                _, loss_value, step,acc,weight1,weight2,weight3,weight4= sess.run([train_step, loss, global_step,accuracy,w1,w2,w3,w4],feed_dict={x: example, y_: label})
                if step % 100 == 0:
                    #ofs.write('weight1' + str(weight1[1])+'\n')
                    #ofs.write('weight2' + str(weight2[1])+'\n')
                    #ofs.write('weight3' + str(weight3[1])+'\n')
                    ofs.write('weight4' + str(weight4[1])+'\n')
                    print("After %d training step(s), loss on training batch is %g,accuracy is %a" % (
                    step, loss_value, acc))
        except tf.errors.OutOfRangeError:
            print('Done training – epoch limit reached')
        finally:
            coord.request_stop()
            ofs.close()
            saver.save(sess, 'net/my_net.ckpt')



example_batch_2, label_batch_2=load_batch([file_name_string],BATCH_SIZE_2)
def get_data():
    saver = tf.train.Saver()
    m, n = 0, 0
    result =np.arange(10)
    ofs = open("./pre.txt", 'w')
    with tf.Session() as sess:
        saver.restore(sess, 'net/my_net.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for i in range(EPOCH_2):
                example, label = sess.run([example_batch_2, label_batch_2])
                _, corr, step,acc,y_pre= sess.run([train_step, correct_prediction, global_step,accuracy,y],feed_dict={x: example, y_: label})
                y_pre = sess.run(tf.nn.softmax(y_pre))
                ofs.write(str(step)+'======'+'y_predict:' + str(y_pre)+'\n')
                if corr == 0:
                    m = m + 1
                if max(y_pre[0]) <= 0.8:
                    n = n + 1
                if max(y_pre[0]) <= 0.8 or corr == 0:
                    res = np.c_[np.array(example), np.array([label])]
                    result = np.row_stack((result, res[0]))
            print(m,'========',n)
        except tf.errors.OutOfRangeError:
            print('Done training – epoch limit reached')
        finally:
            coord.request_stop()
            ofs.close()
            np.savetxt('Ambiguous_data.csv', result, delimiter=',')


def main(argv=None):
    train()
    get_data()


if __name__ == '__main__':
    main()