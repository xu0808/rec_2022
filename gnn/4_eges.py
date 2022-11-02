# -*- coding: utf-8 -*-
# EGES模型
# 改自张朝阳
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam


class EGES(Model):
    def __init__(self, item_num, side_info_nums, vec_dim):
        super(EGES, self).__init__()
        # 节点数
        self.itemNum = item_num
        # 边信息各特征字典大小数组
        self.side_info_count = len(side_info_nums)
        # 节点编号embedding层
        self.id_embed = Embedding(item_num, vec_dim)
        # 边信息embedding层
        self.side_embeds = []
        for num in side_info_nums:
            # 边信息id embedding层
            self.side_embeds.append(Embedding(num, vec_dim))
        # 权重矩阵Embedding层
        self.w = Embedding(item_num, 1 + self.side_info_count)

    # 节点聚合
    def item_agg(self, info_i):
        # 第0维上是batch_size
        item_id = info_i[:, 0]
        # 属性的重要度 a.shape =  (None, 3)
        a = self.w(item_id)
        print('a.shape = ', a.shape)
        # 取softmax，确保权重大于0 weight.shape =  (None, 3, 1)
        weight = tf.expand_dims(tf.nn.softmax(a, axis=0), axis=-1)
        print('weight.shape = ', weight.shape)
        # id_vector.shape =  (None, 1, 16)
        id_vector = tf.expand_dims(self.id_embed(item_id), axis=1)
        print('id_vector.shape = ', id_vector.shape)
        vectors = [id_vector]
        for i in range(1, self.side_info_count + 1):
            # side_vector.shape =  (None, 1, 16)
            side_vector = tf.expand_dims(self.side_embeds[i - 1](info_i[:, i]),
                                         axis=1)
            print('side_vector.shape = ', side_vector.shape)
            vectors.append(side_vector)
        # 向量拼接 vectors.shape =  (None, 3, 16)
        vectors = tf.concat(vectors, axis=1)
        print('vectors.shape = ', vectors.shape)
        # 各向量加权和 vector.shape =  (None, 16)
        vector = tf.reduce_sum(tf.multiply(weight, vectors), axis=1)
        print('vector.shape = ', vector.shape)
        return vector

    def call(self, inputs, training=None, mask=None):
        info_i = inputs[:, :1 + self.side_info_count]
        info_j = inputs[:, 1 + self.side_info_count:]
        vector_i = self.item_agg(info_i)
        vector_j = self.item_agg(info_j)
        # 条件概率及损失函数参考LINE里的二阶相似度 out_put.shape =  (None, 1)
        out_put = tf.sigmoid(tf.reduce_sum(tf.multiply(vector_i, vector_j), axis=-1, keepdims=True))
        print('out_put.shape = ', out_put.shape)
        return out_put


def loss(y_true, y_pred):
    # 负采样损失函数简化版(基于kl散度计算公式)
    return -tf.math.log(tf.sigmoid(y_true * y_pred))


def train(feature, label, item_num, side_info_nums, vec_dim):
    model = EGES(item_num, side_info_nums, vec_dim)
    optimizer = Adam(learning_rate=1e-4)
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(feature, label, batch_size=16, epochs=10)
    return model


if __name__ == "__main__":
    # 节点10000个
    itemNum = 10000
    # 边信息1为10000个,边信息2为30000个
    sideInfoNum = [10000, 30000]
    # 生成向量16位
    embedding_dim = 16

    # 生成样本
    # 样本总数1000个
    total_sample = 1000
    # 特征数据 item_i,item_j
    x = np.random.randint(low=0, high=itemNum - 1, size=(total_sample, 1))
    for cnt in sideInfoNum:
        x = np.hstack([x, np.random.randint(low=0, high=cnt - 1, size=(total_sample, 1))])
    x = np.hstack([x, np.random.randint(low=0, high=itemNum - 1, size=(total_sample, 1))])
    for cnt in sideInfoNum:
        x = np.hstack([x, np.random.randint(low=0, high=cnt - 1, size=(total_sample, 1))])
    y = np.random.randn(total_sample, 1)
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    # 标签数据 item_i,item_j是否被一起点击
    y = np.where(y >= 0, 1, -1)
    y = tf.convert_to_tensor(y, dtype=tf.float64)

    # 模型训练
    eges_model = train(x, y, itemNum, sideInfoNum, embedding_dim)
    # 8、3、5 对应个完整的物品特征
    item = tf.convert_to_tensor([[8, 3, 5]], dtype=tf.float64)
    item_vector = eges_model.item_agg(item)
    print(item_vector)
