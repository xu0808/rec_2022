# -*- coding: utf-8 -*-
# 使用Embedding进行向量分解

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
import reader_2
from utils import sim_top_n


class matrix_cf(Model):
    def __init__(self, user_num, item_num, vec_dim=10):
        super(matrix_cf, self).__init__()
        # 用户数
        self.user_num = user_num
        # 物品数
        self.item_num = item_num
        # 向量维度数
        self.vec_dim = vec_dim
        self.user_embed = Embedding(user_num, vec_dim)
        self.item_embed = Embedding(item_num, vec_dim)

    def call(self, inputs):
        user_embed = self.user_embed(inputs[:, 0])
        print('user_embed.shape = ', user_embed.shape)
        item_embed = self.item_embed(inputs[:, 1])
        print('item_embed.shape = ', item_embed.shape)
        # 对应元素相乘
        m = tf.multiply(user_embed, item_embed)
        print('m.shape = ', m.shape)
        y_hat = tf.reduce_sum(m, axis=1)
        print('y_hat.shape = ', y_hat.shape)
        return y_hat


def loss(label, y_pred):
    # 负采样损失函数简化版(基于kl散度计算公式)
    return (label - y_pred)**2


def train(feature, label, user_num, item_num, vec_dim):
    model = matrix_cf(user_num, item_num, vec_dim)
    optimizer = Adam(learning_rate=1e-4)
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(feature, label, batch_size=16, epochs=100)
    return model


if __name__ == "__main__":
    # 特征数据
    samples, user_list, item_list = reader_2.mf_feature()

    x = np.array(samples)[:, 0:-1]
    y = np.array(samples)[:, -1]
    user_number = len(user_list)
    item_number = len(item_list)
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)

    # 模型训练
    mf_model = train(x, y, user_number, item_number, 10)

    # 向量读取
    user_emb_dict = {user: list(mf_model.user_embed(index).numpy()) for index, user in enumerate(user_list)}
    item_emb_dict = {item: list(mf_model.item_embed(index).numpy()) for index, item in enumerate(item_list)}

    # 向量写入文件
    reader_2.write_sim_dict('3_mf_user_emb', data=user_emb_dict)
    reader_2.write_sim_dict('3_mf_item_emb', data=item_emb_dict)

    # faiss距离计算
    # 2-1、i2i
    item_embs = [item_emb_dict[item] for item in item_list]
    sim_dic_i2i = sim_top_n.sim_i2i(item_list, item_embs)
    reader_2.write_sim_dict('3_mf_i2i', data=sim_dic_i2i)
    # # 2-2、u2i
    user_embs = [user_emb_dict[user] for user in user_list]
    sim_dic_u2i = sim_top_n.sim_u2i(item_list, item_embs, user_list, user_embs)
    reader_2.write_sim_dict('3_mf_u2i', data=sim_dic_u2i)
