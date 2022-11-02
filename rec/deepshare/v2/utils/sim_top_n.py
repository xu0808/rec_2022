#!/usr/bin/env.txt python
# coding: utf-8
# 相似度topN计算

import numpy as np
# pip install faiss-cpu
import faiss

np.printoptions()


def distance_2_sim(distance):
    max_d, min_d = np.max(distance), np.min(distance)
    print('max_d = ', max_d, 'min_d = ', min_d)
    sim = 1 - ((1 + distance - min_d) / (2 + max_d - min_d)) ** 0.5
    return sim


def sim_i2i(keys, item_emb, top_n=5):
    """
    keys item主键列表
    item_emb item向量列表
    top_n item默认取top5
    """
    emb_i = np.array(item_emb).astype(np.float32)
    index_i2i = faiss.IndexFlatL2(emb_i.shape[1])
    index_i2i.add(emb_i)
    # d top5的距离数组，i top5的索引数组
    # 基础验证最近的为自己距离为0
    d, i = index_i2i.search(emb_i, top_n + 1)
    sim = distance_2_sim(d)
    sim_dic_i2i = {}
    for index in range(i.shape[0]):
        top_item = np.array(keys)[i[index]]
        # 第一个为自己，后面为近邻
        sim_dic_i2i[top_item[0]] = [(ind, s) for ind, s in zip(top_item[1:], sim[index, 1:])]
    return sim_dic_i2i


def sim_u2i(user_keys, user_emb, item_keys, item_emb, top_n=5):
    """
    user_keys 用户主键列表
    user_emb  用户向量列表
    item_keys item主键列表
    item_emb item向量列表
    top_n item默认取top5
    """
    emb_i = np.array(item_emb).astype(np.float32)
    index_u2i = faiss.IndexFlatL2(emb_i.shape[1])
    index_u2i.add(emb_i)
    emb_u = np.array(user_emb).astype(np.float32)
    # 为每个用户查找5个电影
    d, i = index_u2i.search(emb_u, top_n)
    sim = distance_2_sim(d)
    sim_dic_u2i = {}
    for index in range(i.shape[0]):
        top_item = np.array(item_keys)[i[index]]
        sim_dic_u2i[user_keys[index]] = [(ind, s) for ind, s in zip(top_item, sim[index])]
    return sim_dic_u2i


if __name__ == '__main__':
    dp = np.array([[1, 2], [3, 4]])
    print('dp = ', dp)


