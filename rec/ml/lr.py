# -*- coding:utf-8 -*-
# GBDT+LR
# https://blog.csdn.net/weixin_42691585/article/details/109337381
# 1. 逻辑回归模型： 连续特征要归一化处理， 离散特征需要one-hot处理

import reader
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')


# 逻辑回归建模
def lr_model(data):
    # 连续特征归一化
    scaler = MinMaxScaler()
    for col in reader.continuous_cols:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    # 离散特征one-hot编码
    for col in reader.category_cols:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    # 把训练集和测试集分开
    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    # 建立模型
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # −(ylog(p)+(1−y)log(1−p)) log_loss
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('tr_logloss: ', tr_logloss)
    print('val_logloss: ', val_logloss)

    # 模型预测
    # n行k列的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率, 这里的1表示点击的概率
    y_pred = lr.predict_proba(test)[:, 1]
    # 这里看前10个， 预测为点击的概率
    print('predict: ', y_pred[:10])


if __name__ == '__main__':
    data0 = reader.load()
    print('type = ', type(data0))
    print('data0.shape = ', data0.shape)
    # 训练和预测lr模型
    lr_model(data0)
