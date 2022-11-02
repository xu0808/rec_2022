#!/usr/bin/env.txt python
# coding: utf-8
# 推理服务
"""
1、召回
i2i
取所有movie_id(hashcode)对应的embedding，与所有的movie_id计算相似后返回topK
u2i:
取所有user_id(hashcode)对应的embedding，与所有的movie_id计算相似后返回topK

2、排序
取当前user_id(hashcode)对应的embedding，与召回候选集的movie_id计算相似后返回topK
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from utils import tf_record
import reader
from utils import sim_top_n

np.printoptions()

if __name__ == "__main__":
    # 1、读取向量
    emb_file = os.path.join(reader.data_dir, 'rating_emb.csv')
    emb_df = pd.read_csv(emb_file)
    print(emb_df[0:5])
    keys, values = emb_df.values[:, 0], emb_df.values[:, 1]
    vecs = []
    for value in list(values):
        value = value[1:-1].replace('\n', '')
        vec = []
        for v in value.split(' '):
            # 小数点、负数、科学计数法
            if v.replace('.', '').replace('e', '').replace('-', '').isnumeric():
                vec.append(float(v))
        vecs.append(vec)
    emb_dict = {k: v for (k, v) in zip(keys, vecs)}

    # 分离用户和电影向量
    keys = ['user_id', 'movie_id', 'rating']
    types = [tf.int64, tf.int64, tf.int64]
    # 分批读出每个特征
    data_set = tf_record.read('rating', keys, types, batch_size=200)
    user_key, user_emb = [], []
    movie_key, movie_emb = [], []
    for user_id, movie_id, _ in data_set:
        for key in user_id.numpy():
            if key in user_key:
                continue
            else:
                user_key.append(key)
                user_emb.append(emb_dict[key])
        for key in movie_id.numpy():
            if key in movie_key:
                continue
            else:
                movie_key.append(key)
                movie_emb.append(emb_dict[key])

    # 2、近邻查找
    # 2-1、i2i
    sim_dic_i2i = sim_top_n.sim_i2i(movie_key, movie_emb)
    # 2-2、u2i
    sim_dic_u2i = sim_top_n.sim_u2i(user_key, user_emb, movie_key, movie_emb)
    print('i2i and u2i finish!')
