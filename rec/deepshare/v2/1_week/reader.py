# coding: utf-8
# 读取评分数据集
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import config
from utils import tf_record

data_dir = os.path.join(config.data_dir, '1')


def read_rating():
    # 数据文件（前10000行)
    start = time.time()
    ratings_file = os.path.join(data_dir, 'ratings10000.dat')
    rating_df0 = pd.read_csv(ratings_file, encoding="utf-8", header=None, sep=r'::')
    print('read dat file use time:%.3f s' % (time.time() - start))
    # 取出数组
    rating_0 = rating_df0.values.astype(np.int32)
    print('rating_0.shape:', rating_0.shape)
    print('rating_0 top5:\r\n', rating_0[:5, :])
    # 取出user_id， movie_id， rating
    rating = rating_0[:, [0, 1, 2]]
    print('rating.shape:', rating.shape)
    # 只取出用户小于2000的记录
    rating_1 = rating[rating[:, 0] < 2000]
    print('rating_1.shape:', rating_1.shape)
    print('rating_1 top5:\r\n', rating_1[:5, :])
    # 离散特征hash化:a.更好的稀疏表示;b.具有一定的特征压缩;c.能够方便模型的部署
    rating_hash = []
    for i in range(rating_1.shape[0]):
        hash_line = [hash('user_id=%d' % rating_1[i, 0]), hash('movie_id=%d' % rating_1[i, 1]), rating_1[i, 2]]
        rating_hash.append(hash_line)
    print('rating_hash.shape: (%d, %d)' % (len(rating_hash), len(rating_hash[0])))
    print('rating_hash last 5:\r\n', np.asarray(rating_hash[-5:]))
    print('read data use time:%.3f s' % (time.time() - start))
    return rating_hash


def write_record(ratings_data):
    start = time.time()
    keys = ['user_id', 'movie_id', 'rating']
    types = ['int64', 'int64', 'int64']
    tf_record.write('rating', keys, types, ratings_data)
    print('write tf record use time:%.3f s' % (time.time() - start))


def read_record():
    start = time.time()
    keys = ['user_id', 'movie_id', 'rating']
    types = [tf.int64, tf.int64, tf.int64]
    # 分批读出每个特征
    data_set = tf_record.read('rating', keys, types)
    data_total = 0
    batch_num = 0
    for user_id, movie_id, rating in data_set:
        if batch_num == 0:
            print('user_id top2 = ', user_id.numpy()[:2])
            print('movie_id top2 = ', movie_id.numpy()[:2])
            print('rating top2 = ', rating.numpy()[:2])
            batch_size = user_id.shape[0]
        batch_num += 1
        data_total += user_id.shape[0]

    # 样本257488，每批200，共1288批
    print('data_set batch_size = ', batch_size)
    print('data_set batch_num = ', batch_num)
    print('data_set data_total = ', data_total)
    print('read tf record use time:%.3f s' % (time.time() - start))


if __name__ == '__main__':
    # data = read_rating()
    # write_record(data)
    read_record()

