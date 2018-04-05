# -*- coding:utf-8 -*-

# 头文件
import numpy as np
import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.chdir("/home/tonny-ftt/PycharmProjects/data_9_feature")


def load_data():
    GM = np.loadtxt('GM.csv',skiprows=1,delimiter=',')
    Adware = np.loadtxt('adware.csv',skiprows=1,delimiter=',')
    Begin = np.loadtxt('begin.csv',skiprows=1,delimiter=',')
    np.random.shuffle(Adware)
    np.random.shuffle(Begin)
    train=np.row_stack((Adware[:12000],Begin[:40000]))
    test=np.row_stack((Adware[12000:15000],Begin[40000:50000]))
    train = np.row_stack((train, GM[:4000]))
    test = np.row_stack((test,GM[4000:4575]))
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.savetxt('train.csv', train, delimiter=',',fmt='%d')
    np.savetxt('test.csv',test,delimiter=',',fmt='%d')



def read_my_file_format(filename_queue):
    reader = tf.TextLineReader()
    key, record_string = reader.read(filename_queue)
    record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1]]
    '''
    record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                     [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                     [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                     [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                     [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                     [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1]]
    '''
    example = tf.decode_csv(record_string,record_defaults=record_defaults)
    return example[:-1], example[-1]




def load_batch_1(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)
    example, label = read_my_file_format(filename_queue)
    min_after_dequeue = 10*batch_size
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch




def load_batch_2(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)
    example, label = read_my_file_format(filename_queue)
    min_after_dequeue = 0
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch




'''
BATCH_SIZE=200
a=tf.Variable(tf.truncated_normal([1,10],stddev=0.1))
example_batch, label_batch=load_batch([file_name_string],BATCH_SIZE)
init=tf.global_variables_initializer()

def train():
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver(max_to_keep=1)
    # 训练模型
    ofs = open("./log.txt", 'w')
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for i in range(10):
                example, label=sess.run([example_batch, label_batch])
                print(label)
        except tf.errors.OutOfRangeError:
            print('Done training – epoch limit reached')
        finally:
            coord.request_stop()
            ofs.close()
            saver.save(sess, 'net/my_net.ckpt')

'''

def main():
    load_data()


if __name__ == '__main__':
    main()

