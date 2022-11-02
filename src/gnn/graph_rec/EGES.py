# -*- coding: utf-8 -*-
# @Time    : 2021/10/31 9:54 PM
# @Author  : zhangchaoyang
# @File    : GESI.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.optimizers import Adam


class GESI(Model):
    def __init__(self, itemNum, sideInfoNum, embedding_dim):
        super(GESI, self).__init__()
        self.itemNum = itemNum
        self.sideInfoCount = len(sideInfoNum)
        self.item_embed = Embedding(itemNum, embedding_dim)
        self.side_embeds = []
        for num in sideInfoNum:
            self.side_embeds.append(Embedding(num, embedding_dim))
        self.W = Embedding(itemNum, 1 + self.sideInfoCount)  # 权重矩阵用Embedding层来实现

    def weighted_pooling(self, info_i):
        item_id = info_i[:, 0]  # info_i第0维上是batch_size
        a = self.W(item_id)  # 取得该商品上各个属性的重要度
        weight = tf.expand_dims(tf.nn.softmax(a, axis=0), axis=-1)  # 取softmax，确保权重大于0

        item_vector = tf.expand_dims(self.item_embed(item_id), axis=1)
        vectors = [item_vector]
        for i in range(1, self.sideInfoCount + 1):
            side_vector = tf.expand_dims(self.side_embeds[i - 1](info_i[:, i]),
                                         axis=1)
            vectors.append(side_vector)
        vectors = tf.concat(vectors, axis=1)  # 各个属性的向量

        sum = tf.reduce_sum(tf.multiply(weight, vectors), axis=1)  # 对各属性的向量求加权和
        return sum

    def call(self, inputs, training=None, mask=None):
        info_i = inputs[:, :1 + self.sideInfoCount]
        info_j = inputs[:, 1 + self.sideInfoCount:]
        vector_i = self.weighted_pooling(info_i)
        vector_j = self.weighted_pooling(info_j)
        cond_prob = tf.sigmoid(tf.reduce_sum(vector_i * vector_j, axis=-1, keepdims=True))  # 条件概率及损失函数参考LINE里的二阶相似度
        return cond_prob


def train_GESI(x, y, itemNum, sideInfoNum, embedding_dim):
    model = GESI(itemNum, sideInfoNum, embedding_dim)
    optimizer = Adam(learning_rate=1e-4)  # 优化算法
    model.compile(loss=kl_dist, optimizer=optimizer)
    model.fit(x, y, batch_size=16, epochs=3)
    return model


def kl_dist(y_true, y_pred):
    return -tf.math.log(tf.sigmoid(y_true * y_pred))


if __name__ == "__main__":
    itemNum = 10000
    sideInfoNum = [10000, 30000]
    total_sample = 1000
    embedding_dim = 100
    x = np.random.randint(low=0, high=itemNum - 1, size=(total_sample, 1))
    for cnt in sideInfoNum:
        x = np.hstack([x, np.random.randint(low=0, high=cnt - 1, size=(total_sample, 1))])
    x = np.hstack([x, np.random.randint(low=0, high=itemNum - 1, size=(total_sample, 1))])
    for cnt in sideInfoNum:
        x = np.hstack([x, np.random.randint(low=0, high=cnt - 1, size=(total_sample, 1))])
    y = np.random.randn(total_sample, 1)
    y = np.where(y >= 0, 1, -1)

    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    model = train_GESI(x, y, itemNum, sideInfoNum, embedding_dim)
    unseen = tf.convert_to_tensor([[8, 3, 5]], dtype=tf.float64)
    vector = model.weighted_pooling(unseen)
    print(vector)
