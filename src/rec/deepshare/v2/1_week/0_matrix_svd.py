#!/usr/bin/env.txt python
# coding: utf-8
# 基于奇异值分解SVD以及协同过滤在推荐系统的应用
# https://blog.csdn.net/Jayphone17/article/details/113106982
# 奇异值分解SVD原理及应用
# https://blog.csdn.net/fengdu78/article/details/122852671
"""
核心三点:
1、衡量物品之间的相似性
2、评分估计
3、稀疏评分矩阵的处理
# 奇异值分解中取前k个特征值:取决于至少需要多少个奇异值的平方和才能达到所有平方和的90%
"""

import numpy as np
import tensorflow as tf


def np_svd(mat):
    u, sigma, v = np.linalg.svd(mat)
    print('u = ', u)
    print('sigma = ', sigma)
    print('v = ', v)
    s_1 = np.column_stack((np.diag(sigma), np.zeros(sigma.shape[0])))
    mat_1 = u.dot(s_1.dot(v))
    print('mat = ', mat)
    print('mat_1 = ', mat_1)
    # 奇异值重要性计算
    total = float(np.sum(sigma ** 2))
    sum_sub = 0
    for k in range(len(sigma)):
        sum_sub = sum_sub + sigma[k] * sigma[k]
        print('sigma[%d] = %.3f' % (k + 1, float(sum_sub) / total))


def tf_svd(mat):
    mat_tf = tf.constant(mat, dtype=tf.float32)
    sigma, u, v = tf.linalg.svd(mat_tf)
    print('u = ', u.numpy())
    print('sigma = ', sigma.numpy())
    print('v = ', v.numpy())
    s_1 = np.column_stack((np.diag(sigma.numpy()), np.zeros(sigma.numpy().shape[0])))
    mat_1 = u.numpy().dot(s_1.dot(v))
    print('mat = ', mat)
    print('mat_1 = ', mat_1)


def svd():
    row, rank, column = 3, 2, 4
    u_0 = np.random.random(size=(row, rank))
    v_0 = np.random.random(size=(rank, column))
    matrix = np.dot(u_0, v_0)
    # 1、np svd
    np_svd(matrix)
    # # 2、tf svd
    tf_svd(matrix)


def sim(vector_1, vector_2):
    dot_prod = float(np.dot(vector_1.T, vector_2))
    norm_prod = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    return 0.5 + 0.5 * (dot_prod / norm_prod)


def score(data, data_rc, user_index, item_index):
    # 物品总数
    n = np.shape(data)[1]
    score_mutiple_sum = 0
    # 加权相似度之和
    score_rc_sum = 0
    # itemIndex菜品与其他菜品两两相似度之和
    for i in range(n):
        # 评分压阵中的相似度
        score_0 = data[user_index, i]
        if score_0 == 0 or i == item_index:
            continue
        # 评分压缩矩阵中的相似度
        score_rc = sim(data_rc[:, i], data_rc[:, item_index])
        # 利用SVD后的矩阵
        # itemIndex与第i个物品的相似度
        score_rc_sum += score_rc
        score_mutiple_sum += score_0 * score_rc
    if score_rc_sum == 0:
        return 0
    return score_mutiple_sum/score_rc_sum


def rec():
    # 评分数据
    data = np.array([[4, 2, 2, 5, 1, 4, 5, 1, 1, 3, 3],
                     [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
                     [4, 0, 4, 4, 0, 3, 5, 4, 0, 3, 4],
                     [3, 0, 2, 4, 0, 3, 5, 0, 2, 2, 3],
                     [0, 0, 2, 0, 0, 5, 0, 0, 0, 5, 3],
                     [4, 0, 1, 4, 1, 1, 2, 0, 1, 1, 1],
                     [0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 5],
                     [0, 0, 5, 2, 2, 4, 4, 5, 2, 2, 0],
                     [3, 0, 0, 3, 4, 5, 5, 3, 4, 3, 5],
                     [3, 5, 5, 3, 5, 2, 4, 3, 1, 2, 5],
                     [4, 3, 5, 5, 5, 5, 5, 5, 5, 5, 3],
                     [0, 0, 5, 1, 0, 5, 5, 0, 3, 1, 4],
                     [4, 0, 4, 4, 1, 4, 5, 4, 5, 2, 4],
                     [4, 0, 3, 3.5, 0, 4, 4, 0, 0, 2, 4],
                     [3, 0, 5, 5, 0, 4, 5, 5, 5, 4, 5],
                     [0, 4, 3, 2, 3, 5, 3, 0, 2, 3, 5],
                     [5, 0, 4, 5, 0, 4, 4, 5, 3, 3.5, 5],
                     [4, 0, 4.5, 5, 0, 3, 4, 2, 2, 3, 4],
                     [3, 3, 5, 4, 4, 5, 5, 4, 5, 1, 5],
                     [4, 3, 5, 3, 3, 5, 4, 4, 3, 4, 5],
                     [2, 3, 4, 2, 0, 3.5, 3.5, 1, 1, 1.5, 4],
                     [2, 0, 3, 3, 3, 3, 5, 2, 2, 4, 5],
                     [4, 4, 4, 5, 0, 2, 4, 3, 2, 1, 5],
                     [1, 0, 1, 1, 0, 1, 2, 1, 3, 0, 3],
                     [3, 0, 5, 3, 3, 5, 3, 3, 2, 3, 5],
                     [5, 0, 5, 3, 3, 2, 5, 0, 5, 2, 1],
                     [4, 0, 0, 4, 0, 3, 4, 0, 4, 3, 5],
                     [2, 0, 5, 4, 5, 1, 5, 2, 3, 2, 0]])
    print('data.shape = ', data.shape)
    # 奇异值sigma获取特征重要度
    u, sigma, vt = np.linalg.svd(data)
    print('sigma = ', sigma)
    total = float(np.sum(sigma ** 2))
    sigma_sum = 0
    k_num = 0
    # 奇异值分解中取前k个特征值:取决于至少需要多少个奇异值的平方和才能达到所有平方和的90%
    for k in range(len(sigma)):
        sigma_sum = sigma_sum + sigma[k] * sigma[k]
        if float(sigma_sum) / total > 0.9:
            k_num = k + 1
            break
    sigma_k = np.mat(np.eye(k_num) * sigma[:k_num])
    print('sigma_k.shape = ', sigma_k.shape)
    print('sigma_k = ', sigma_k)
    # 评分压缩矩阵
    data_rc = sigma_k * u.T[:k_num, :] * data
    print('data_rc.shape = ', data_rc.shape)
    print('data_rc = ', data_rc)
    n = np.shape(data)[1]
    user_index = 25
    for i in range(n):
        user_score = data[user_index, i]
        if user_score != 0:
            continue
        print("index:{},score:{}".format(i, score(data, data_rc, user_index, i)))


if __name__ == "__main__":
    rec()
