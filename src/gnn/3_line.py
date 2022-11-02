#!/usr/bin/env.txt python
# coding: utf-8
# LINE
#
# DeepWalk、Node2vec、LINE
# https://www.cnblogs.com/Lee-yl/p/12670515.html
"""
与DeepWalk使用DFS构造邻域不同的是，LINE可以看作是一种使用BFS构造邻域的算法。
此外，LINE还可以应用在(有向、无向亦或是有权重)图中(DeepWalk仅能用于无权图)，且对图中顶点之间的相似度的定义不同。
"""

import math
import random
import numpy as np
import networkx as nx

# tf2的相关模块
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda
from tensorflow.python.keras.models import Model

# alias算法的构造表和采样
from alias import create_alias_table, alias_sample
import os


# 节点字典
def node_dict(graph):
    node2idx = {}
    idx2node = []
    for node in graph.nodes():
        node2idx[node] = len(idx2node)
        idx2node.append(node)
    return idx2node, node2idx


# 损失函数(KL散度)
def kl_loss(y_true, y_pred):
    return -K.mean(K.log(K.sigmoid(y_true * y_pred)))


# 创建模型
def create_model(node_num, vec_size, order='second'):
    v_i = Input(shape=(1,))
    v_j = Input(shape=(1,))

    first_emb = Embedding(node_num, vec_size, name='first_emb')
    second_emb = Embedding(node_num, vec_size, name='second_emb')
    context_emb = Embedding(node_num, vec_size, name='context_emb')

    v_i_emb = first_emb(v_i)
    v_j_emb = first_emb(v_j)
    v_i_emb_second = second_emb(v_i)
    v_j_context_emb = context_emb(v_j)

    # Lambda函数，Lambda（function）（tensor)
    first = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=-1, keepdims=False), name='first_order')([v_i_emb, v_j_emb])
    second = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=-1, keepdims=False), name='second_order')([v_i_emb_second, v_j_context_emb])

    if order == 'first': output_list = [first]
    elif order == 'second': output_list = [second]
    else: output_list = [first, second]
    model = Model(inputs=[v_i, v_j], outputs=output_list)
    return model, {'first': first_emb, 'second': second_emb}


# LINE模型类
class LINE:
    def __init__(self, graph, vec_size=8, negative_ratio=5, order='second'):
        """
        :param graph:
        :param vec_size:
        :param negative_ratio:
        :param order: 'first','second','all'
        """
        if order not in ['first', 'second', 'all']:
            raise ValueError('mode must be fisrt,second,or all')

        self.graph = graph
        self.idx2node, self.node2idx = node_dict(graph)
        self.use_alias = True
        self.vec_size = vec_size
        self.order = order
        self._embeddings = {}
        self.negative_ratio = negative_ratio
        self.order = order
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.samples_per_epoch = self.edge_size * (1 + negative_ratio)

        # 采样表，获取边和顶点的采样accept和alias
        self._gen_sampling_table()
        # 建立模型，执行create_model和batch_iter
        self.reset_model()

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = ((self.samples_per_epoch - 1) // self.batch_size + 1) * times

    def reset_model(self, opt='adam'):
        self.model, self.embedding_dict = create_model(self.node_size, self.vec_size, self.order)
        self.model.compile(opt, kl_loss)
        self.batch_it = self.batch_iter(self.node2idx)

    def _gen_sampling_table(self):
        # create sampling table for vertex
        power = 0.75
        node_degree = np.zeros(self.node_size)  # out degree
        node2idx = self.node2idx
        for edge in self.graph.edges():
            node_degree[node2idx[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight', 1.0)

        total_sum = sum([math.pow(node_degree[i], power) for i in range(self.node_size)])
        norm_prob = [float(math.pow(node_degree[j], power)) /total_sum for j in range(self.node_size)]
        self.node_accept, self.node_alias = create_alias_table(norm_prob)

        # create sampling table for edge
        numEdges = self.graph.number_of_edges()
        total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0) for edge in self.graph.edges()])
        norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) * numEdges/total_sum for edge in self.graph.edges()]
        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)

    def batch_iter(self, node2idx):
        edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph.edges()]
        data_size = self.graph.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)
        while True:
            if mod == 0:
                h = []
                t = []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                sign = np.ones(len(h))
            else:
                sign = np.ones(len(h)) * -1
                t = []
                for i in range(len(h)):
                    t.append(alias_sample(self.node_accept, self.node_alias))

            if self.order == 'all':
                yield ([np.array(h), np.array(t)], [sign, sign])
            else:
                yield ([np.array(h), np.array(t)], [sign])
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index >= data_size:
                count += 1
                mod = 0
                h = []
                shuffle_indices = np.random.permutation(np.arange(data_size))
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)

    def get_embeddings(self, ):
        self._embeddings = {}
        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[0], self.embedding_dict['second'].get_weights()[0]))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding

        return self._embeddings

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        self.reset_training_config(batch_size, times)
        hist = self.model.fit_generator(self.batch_it, epochs=epochs, initial_epoch=initial_epoch,
                                        steps_per_epoch=self.steps_per_epoch, verbose=verbose)

        return hist


if __name__ == "__main__":
    # gnn数据根目录
    gnn_base_dir = 'C:\\work\\data\\gnn\\base'
    wiki_edge_file = os.path.join(gnn_base_dir, 'Wiki_edgelist.txt')
    # 加载图数据
    G = nx.read_edgelist(wiki_edge_file,  create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    # node number =  2405 => edge number =  16523
    print('node number = ', G.number_of_nodes(), '=> edge number = ', G.number_of_edges())
    # LINE模型训练
    model = LINE(G, vec_size=16, order='second')
    model.train(batch_size=1024, epochs=50, verbose=2)
    emb = model.get_embeddings()
    # 打印前3个
    print('emb size = ', len(emb))
    i = 0
    for key in emb.keys():
        print('key = ', key, '=> value = ', emb[key])
        i += 1
        if i > 3: break