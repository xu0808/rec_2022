# -*- coding: utf-8 -*-
# 使用Embedding进行向量分解

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from utils import sim_top_n
import reader_2

batch_size = 200


# keys = ['user_id', 'article_id', 'environment', 'region', 'label']
# types = [tf.int64, tf.int64, tf.int64, tf.int64, tf.float32]
# # 分批读出每个特征
# data_set = tf_record.read('click_log', keys, types, batch_size=batch_size)


class fm(Model):
    def __init__(self, user_num, item_num, dict_sizes, k, vec_dim=16):
        super(fm, self).__init__()

        # 特征向量的总维度
        total_dim = (1 + 1 + len(dict_sizes)) * vec_dim
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(1,), dtype='float32'), trainable=True)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(total_dim, 1), dtype='float32'), trainable=True)

        # 用户数
        self.user_num = user_num
        # 物品数
        self.item_num = item_num
        # 其它特征字典大小列表
        self.dict_sizes = dict_sizes
        # 向量维度数
        self.vec_dim = vec_dim
        self.user_embed = Embedding(user_num, vec_dim)
        self.item_embed = Embedding(item_num, vec_dim)
        # 隐向量
        self.k = self.add_weight(name='k', shape=(total_dim, k), initializer='glorot_uniform', trainable=True)
        # 特征embedding层
        self.feature_embeds = []
        for num in dict_sizes:
            self.feature_embeds.append(Embedding(num, vec_dim))

    def call(self, inputs):
        print('inputs.shape = ', inputs.shape)
        # 所有特征向量
        user_ids, item_ids = inputs[:, 0], inputs[:, 1]
        vectors = [self.user_embed(user_ids), self.item_embed(item_ids)]
        for i, emb in enumerate(self.feature_embeds):
            feature_vector = emb(inputs[:, i + 2])
            print('feature_vector.shape = ', feature_vector.shape)
            vectors.append(feature_vector)

        # 添加中间维度并且拼接
        vector_concat = tf.concat(vectors, axis=1)
        print('vector_concat.shape = ', vector_concat.shape)

        # 线性回归
        liner = tf.matmul(vector_concat, self.w) + self.b
        print('liner.shape = ', liner.shape)

        # FM的二阶交叉项
        inter_1 = tf.pow(tf.matmul(vector_concat, self.k), 2)
        inter_2 = tf.matmul(tf.pow(vector_concat, 2), tf.pow(self.k, 2))
        print('vector_concat.shape = ', vector_concat.shape, 'k.shape = ', self.k.shape)
        print('inter_1.shape = ', inter_1.shape, 'inter_2.shape = ', inter_2.shape)
        cross = tf.reduce_sum(tf.subtract(inter_1, inter_2), 1, keepdims=True) * 0.5
        print('cross.shape = ', cross.shape)

        # 获得FM模型（线性回归 + FM的二阶交叉项）
        y_hat = tf.sigmoid(tf.add(liner, cross))
        print('y_hat.shape = ', y_hat.shape)
        return y_hat


def train(x, y, user_num, item_num, dict_sizes, k=4, vec_dim=16):
    model = fm(user_num, item_num, dict_sizes, k, vec_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.binary_accuracy])
    model.fit(x, y, batch_size=200, epochs=100)
    return model


if __name__ == "__main__":
    # 所有特征字典:user_id,item_id,env,region
    dict_list, samples = reader_2.fm_feature()
    samples = np.array(samples)
    x0 = tf.convert_to_tensor(samples[:, :-1], dtype=tf.int64)
    y0 = tf.convert_to_tensor(samples[:, -1], dtype=tf.float32)
    # 模型训练 字典大小（711、389、3、25）
    user_list, item_list, env_list, region_list = dict_list
    mf_model = train(x0, y0, user_num=len(user_list), item_num=len(item_list), dict_sizes=[len(env_list), len(region_list)])

    # 向量读取
    user_emb_dict = {user: list(mf_model.user_embed(index).numpy()) for index, user in enumerate(user_list)}
    item_emb_dict = {item: list(mf_model.item_embed(index).numpy()) for index, item in enumerate(item_list)}

    # 向量写入文件
    reader_2.write_sim_dict('4_fm_user_emb', data=user_emb_dict)
    reader_2.write_sim_dict('4_fm_item_emb', data=item_emb_dict)

    # faiss距离计算
    # 2-1、i2i
    item_embs = [item_emb_dict[item] for item in item_list]
    sim_dic_i2i = sim_top_n.sim_i2i(item_list, item_embs)
    reader_2.write_sim_dict('4_fm_i2i', data=sim_dic_i2i)
    # # 2-2、u2i
    user_embs = [user_emb_dict[user] for user in user_list]
    sim_dic_u2i = sim_top_n.sim_u2i(item_list, item_embs, user_list, user_embs)
    reader_2.write_sim_dict('4_fm_u2i', data=sim_dic_u2i)
