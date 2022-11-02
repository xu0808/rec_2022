#!/usr/bin/env.txt python
# coding: utf-8
# 数据读取

import os
import pandas as pd
import json
import random
from collections import defaultdict
import numpy as np
from functools import reduce
from utils import config
from utils import tf_record

data_dir = os.path.join(config.data_dir, '2')

features = 'user_id,article_id,environment,region'.split(',')
label = 'label'


def read_log():
    """
    读取用户行为日志
    :return:
     user_item_dict 用户的点击文章列表字典
     item_user_dict  文章的点击用户列表字典
     item_cate_dict  文章分类字典
     user_item_ts   用户点击文章时间字典
    """
    # 文章信息
    articles_file = os.path.join(data_dir, 'articles.csv')
    article_df = pd.read_csv(articles_file, encoding='utf-8')
    print('article_df top5:\n', article_df[:5])
    item_info = article_df.values[:, :2]
    item_cate_dict = {}
    for i in range(item_info.shape[0]):
        [item_id, cate_id] = item_info[i]
        item_cate_dict[item_id] = cate_id

    # 点击日志
    click_log_file = os.path.join(data_dir, 'click_log.csv')
    click_log_df = pd.read_csv(click_log_file, encoding='utf-8')
    print('click_log_df:\n', click_log_df[:5])
    """只取前10000个用户点记录"""
    click_log = click_log_df.values[:10000, :3]
    print('click_log:\n', click_log[0:5])

    # 样本统计
    # 用户点击过的文章列表
    user_item_dict = defaultdict(list)
    # 文章的点击用户列表
    item_user_dict = defaultdict(list)
    # 用户点击文章时间
    user_item_ts = defaultdict(int)
    for i in range(click_log.shape[0]):
        [user_id, item_id, ts] = click_log[i]
        user_item_dict[user_id].append(item_id)
        item_user_dict[item_id].append(user_id)
        user_item_ts['%d_%d' % (user_id, item_id)] = ts

    print('user_num = ', len(user_item_dict), ',item_num = ', len(item_user_dict), ',log_num = ', len(user_item_ts))
    return user_item_dict, item_user_dict, item_cate_dict, user_item_ts


def mf_feature():
    """MF模型特征加工"""
    # 1、行为日志读取
    dict_list = read_log()
    user_item_dict, item_user_dict = dict_list[0], dict_list[1]

    # 2、生成字典
    user_list = list(user_item_dict)
    user_dict = {user: index for index, user in enumerate(user_list)}
    item_list = list(item_user_dict.keys())
    item_dict = {item: index for index, item in enumerate(item_list)}

    # 3、生成样本
    samples = []
    for user, items in user_item_dict.items():
        # 所有的负采样的样本(连续排除)
        possible_negs = reduce(np.setdiff1d, (item_list, items))
        for item in items:
            samples.append([user_dict[user], item_dict[item], 1])
            # 负采样样本
            neg_item = np.random.choice(possible_negs, 1, replace=False)
            samples.append([user_dict[user], item_dict[neg_item[0]], 0])
    return samples, user_list, item_list


def write_sim_dict(name, data={}):
    """相似表字典写入文件"""
    df = pd.DataFrame(columns=['key', 'value'], data=[[k, list(v)] for k, v in data.items()])
    # 数据文件
    data_file = os.path.join(data_dir, '%s.csv' % name)
    df.to_csv(data_file, index=False, sep=',')


def read_sim_file(name):
    # 数据文件
    data_file = os.path.join(data_dir, '%s.csv' % name)
    df = pd.read_csv(data_file, encoding='utf-8')
    # "[(111031, 0.3679)]" -> json解析时需要把元组改成list格式
    values = {k: json.loads(v.replace('(', '[').replace(')', ']')) for k, v in df.values}
    return values


def fm_feature():
    """fm模型特征(负采样)"""
    click_log_file = os.path.join(data_dir, 'click_log.csv')
    click_log_df = pd.read_csv(click_log_file, encoding='utf-8')
    # 只取user_id,article_id,environment,region
    click_log_0 = click_log_df.values.astype(np.int32)
    print('click_log_0.shape:', click_log_0.shape)
    # 只取前2千条行为记录
    click_log_1 = click_log_0[0:2000, [0, 1, 3, 4]]
    print('click_log_1.shape:', click_log_1.shape)
    print('click_log_1:\n', click_log_1[-5:])
    # 所有特征值列表
    list_list = []
    # 所有特征字典:user_id,item_id,env,region
    dict_list = []
    for i in range(4):
        values = list(np.unique(click_log_1[:, i]))
        list_list.append(values)
        name_id_dict = {value: index for index, value in enumerate(values)}
        dict_list.append(name_id_dict)

    # 文章清单
    article_list = list(np.unique(click_log_1[:, 1]))
    samples = []
    for i in range(click_log_1.shape[0]):
        item_id = click_log_1[i][1]
        # 字段转换成索引
        line = [dict_list[j][click_log_1[i][j]] for j in range(4)]
        samples.append(line + [1])
        # 每个样本取10个负样本
        for neg_id in random.sample(article_list, 10):
            if neg_id == item_id:
                continue
            neg_index = dict_list[1][neg_id]
            samples.append([line[0], neg_index, line[2], line[3], 0])
    print('samples:\n', samples[:5])
    return list_list, samples


def write_recod():
    feature_hash = fm_feature()
    keys = features + [label]
    types = ['int64']*4 + ['float']
    tf_record.write('click_log', keys, types, feature_hash)


if __name__ == '__main__':
    # 读取用户行为日志
    # data_0 = read_log()
    # user点击序列写入
    # write_sim_dict('0_user_items', data=data_0[0])
    # 读取相似表
    sims = read_sim_file('1_item_cf_i2i')
    print(sims)
    # 写入fm模型特征到tfrecord
    # write_recod()

