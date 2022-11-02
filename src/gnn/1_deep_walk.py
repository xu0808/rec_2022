#!/usr/bin/env.txt python
# coding: utf-8
# DeepWalk
#
# DeepWalk、Node2vec、LINE
# https://www.cnblogs.com/Lee-yl/p/12670515.html
"""
Random Walk：一种可重复访问已访问节点的深度优先遍历算法。
给定当前访问起始节点，从其邻居中随机采样节点作为下一个访问节点，
重复此过程，直到访问序列长度满足预设条件。
Word2vec：接着利用skip-gram模型进行向量学习。
"""

from joblib import Parallel, delayed
import itertools
import random
from gensim.models import Word2Vec
from walker import RandomWalker
import networkx as nx
import os



class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.walker = RandomWalker(graph)
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter
        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")
        self.w2v_model = model
        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


if __name__ == "__main__":
    # gnn数据根目录
    gnn_base_dir = 'C:\\work\\data\\gnn\\base'
    wiki_edge_file = os.path.join(gnn_base_dir, 'Wiki_edgelist.txt')
    G = nx.read_edgelist(wiki_edge_file, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(embed_size=16, window_size=5, iter=3)
    emb = model.get_embeddings()
    # 打印前3个
    print('emb size = ', len(emb))
    i = 0
    for key in emb.keys():
        print('key = ', key, '=> value = ', emb[key])
        i += 1
        if i > 3: break
