#!/usr/bin/env.txt python
# coding: utf-8
"""
   userCF的主要思想是：给用户推荐与其相似的用户喜欢的的物品。模式：u2u2i
   一、逐个item计算user相似度并累加(用户平均活跃度/(物品点击用户数 + 1))
   二、逐个user计算相似user的topN(sum(用户相似度)/sqrt(用户i活跃度 * 用户j活跃度))
"""
import math
from collections import defaultdict
import reader_2


def user_cf_sim(user_item_dict, item_user_dict, top_n=20):
    """
     计算用户的相似用户topN
     :parma:
     user_item_dict 用户的点击文章列表字典
     item_user_dict  文章的点击用户列表字典
     top_n 相似top数
     :return:
     u2u_sim 用户相似度字典
    """
    # 一、逐个item计算user相似度并累加(用户平均活跃度/(物品点击用户数 + 1))
    u2u_sim_0 = defaultdict(dict)
    for _, users in item_user_dict.items():
        for u in users:
            num_i = len(user_item_dict[u])
            for v in users:
                if u == v:
                    continue
                # 1、用户平均活跃度作为活跃度的权重
                num_j = len(user_item_dict[v])
                act_weight = 0.1 * 0.5 * (num_i + num_j)
                # 2、考虑多种因素的权重计算最终的相似度
                u2u_sim_0[u].setdefault(v, 0)
                u2u_sim_0[u][v] += round(act_weight / math.log(len(users) + 1), 4)

    # 二、逐个user计算相似user的topN(sum(用户相似度)/sqrt(用户i活跃度 * 用户j活跃度))
    u2u_sim = {}
    for u, sims in u2u_sim_0.items():
        user_sims = {}
        for v, sim in sims.items():
            user_sims[v] = round(sim/math.sqrt(len(user_item_dict[u]) * len(user_item_dict[v])), 4)
        # 取相似用户评分topN
        u2u_sim[u] = sorted(user_sims.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:top_n]
    return u2u_sim


if __name__ == '__main__':
    # 1、行为日志读取
    dict_list = reader_2.read_log()
    # 2、item_cf相似物品计算
    u2u_sim_dict = user_cf_sim(dict_list[0], dict_list[1], top_n=20)
    for user in list(u2u_sim_dict.keys())[:5]:
        print('u2u result:', user, u2u_sim_dict[user])

    # 3、user相似表写入
    reader_2.write_sim_dict('2_user_cf_u2u', data=u2u_sim_dict)
