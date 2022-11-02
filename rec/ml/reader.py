# -*- coding:utf-8 -*-
# 数据加载
# 基于kaggle上的一个比赛数据集
# 这个比赛的任务就是：开发预测广告点击率(CTR)的模型。给定一个用户和他正在访问的页面，预测他点击给定广告的概率是多少？

# criteo-Display Advertising Challenge比赛的部分数据集， 里面有train.csv和test.csv两个文件：
# train.csv： 训练集由Criteo 7天内的部分流量组成。每一行对应一个由Criteo提供的显示广告。
#             为了减少数据集的大小，正(点击)和负(未点击)的例子都以不同的比例进行了抽样。示例是按时间顺序排列的
# test.csv: 测试集的计算方法与训练集相同，只是针对训练期之后一天的事件

# 字段说明：
# Label： 目标变量， 0表示未点击， 1表示点击
# l1-l13: 13列的数值特征， 大部分是计数特征
# C1-C26: 26列分类特征， 为了达到匿名的目的， 这些特征的值离散成了32位的数据表示

# 数据导入与简单处理
import os
import pandas as pd

# 连续特征列
continuous_cols = ['I' + str(i + 1) for i in range(13)]
# 离散特征列
category_cols = ['C' + str(i + 1) for i in range(26)]
# 数据文件
data_dir = 'C:\\work\\data\\rec\\ml'

"""数据读取与预处理"""


def load():
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    # 简单的数据预处理
    # 去掉id列， 把测试集和训练集合并， 填充缺失值
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)
    df_test['Label'] = -1
    data = pd.concat([df_train, df_test])
    data.fillna(-1, inplace=True)
    return data


if __name__ == '__main__':
    data0 = load()
    print('type = ', type(data0))
    print('data0.shape = ', data0.shape)
    print(' data.keys() = ', str(data0.keys()))
