#!/usr/bin/env.txt python
# coding: utf-8
# Node2vec
#
# DeepWalk、Node2vec、LINE
# https://www.cnblogs.com/Lee-yl/p/12670515.html
"""
相对于DeepWalk, node2vec的改进主要是对基于随机游走的采样策略的改进
node2vec是结合了BFS和DFS的Deepwalk改进的随机游走算法
"""

from gensim.models import Word2Vec
from walker import RandomWalker
import networkx as nx
import os


class Node2Vec:
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):
        self.graph = graph
        self._embeddings = {}
        # 采样
        self.walker = RandomWalker(graph, p=p, q=q, )

        # 构造别名表
        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        # 采样
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        # Word2vec
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
    model = Node2Vec(G, walk_length=10, num_walks=80, p=0.25, q=4, workers=1)
    model.train(embed_size=16, window_size=5, iter=3)
    emb = model.get_embeddings()
    # 打印前3个
    print('emb size = ', len(emb))
    i = 0
    for key in emb.keys():
        print('key = ', key, '=> value = ', emb[key])
        i += 1
        if i > 3: break