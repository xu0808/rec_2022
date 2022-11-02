#!/usr/bin/env.txt python
# coding: utf-8
"""
    ItemCF的主要思想是：给用户推荐之前喜欢物品的相似物品。模式：i2i
    一、逐个user计算item相似度并累加(时间相似*品类相似/(用户点击物品数 + 1))
    二、逐个item计算相似item的topN(sum(物品相似度)/sqrt(物品i点击数 * 物品j点击数))
"""

import math
import numpy as np
from collections import defaultdict
import reader_2


def item_cf_sim(user_item_dict, item_user_dict, item_cate_dict, user_item_ts, top_n=20):
    """
     计算物品的相似物品topN
     :parma:
     user_item_dict 用户的点击文章列表字典
     item_user_dict  文章的点击用户列表字典
     item_cate_dict  文章分类字典
     user_item_ts   用户点击文章时间字典
     top_n 相似top数
     :return:
     i2i_sim 物品相似度字典
    """
    i2i_sim_0 = defaultdict(dict)
    # 一、逐个user计算item相似度并累加(时间相似*品类相似/(用户点击物品数 + 1))
    for user, items in user_item_dict.items():
        for i in items:
            ts_i = user_item_ts['%d_%d' % (user, i)]
            cat_i = item_cate_dict[i]
            for j in items:
                if i == j:
                    continue
                # 1、点击时间权重,点击时间相近权重大,相远权重小
                ts_j = user_item_ts['%d_%d' % (user, j)]
                ts_weight = np.exp(0.7 ** np.abs(ts_i - ts_j))
                # 2、类别权重,其中类别相同权重大
                cat_j = item_cate_dict[j]
                type_weight = 1.0 if cat_i == cat_j else 0.7
                # 3、考虑多种因素的权重计算最终相似度
                i2i_sim_0[i].setdefault(j, 0)
                i2i_sim_0[i][j] += round(ts_weight * type_weight / math.log(len(items) + 1), 4)

    # 二、逐个item计算相似item的topN(sum(物品相似度)/sqrt(物品i点击数 * 物品j点击数))
    i2i_sim = {}
    for i, sims in i2i_sim_0.items():
        item_sims = {}
        for j, sim in sims.items():
            item_sims[j] = round(sim / math.sqrt(len(item_user_dict[i]) * len(item_user_dict[j])), 4)
        # 取相似item评分topN
        i2i_sim[i] = sorted(item_sims.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:top_n]
    return i2i_sim


if __name__ == '__main__':
    # 1、行为日志读取
    dict_list = reader_2.read_log()
    # 2、item_cf相似物品计算
    i2i_sim_dict = item_cf_sim(dict_list[0], dict_list[1], dict_list[2], dict_list[3], top_n=20)
    for item in list(i2i_sim_dict.keys())[:5]:
        print('i2i result:', item, i2i_sim_dict[item])

    # 3、item相似表写入
    reader_2.write_sim_dict('1_item_cf_i2i', data=i2i_sim_dict)
