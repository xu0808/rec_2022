# -*- coding:utf-8 -*-
# GBDT+LR
# https://blog.csdn.net/weixin_42691585/article/details/109337381
# 2. GBDT模型： 树模型连续特征不需要归一化处理， 但是离散特征需要one-hot处理

import reader
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')


# GBDT建模
def gbdt_model(data):
    # 离散特征one-hot编码
    for col in reader.category_cols:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    # 训练集和测试集分开
    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)
    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    # 建模
    gbm = lgb.LGBMClassifier(boosting_type='gbdt',  # 这里用gbdt
                             objective='binary',
                             subsample=0.8,
                             min_child_weight=0.5,
                             colsample_bytree=0.7,
                             num_leaves=100,
                             max_depth=12,
                             learning_rate=0.01,
                             n_estimators=10000)
    gbm.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_names=['train', 'val'],
            eval_metric='binary_logloss',
            early_stopping_rounds=100)

    # −(ylog(p)+(1−y)log(1−p)) log_loss
    tr_logloss = log_loss(y_train, gbm.predict_proba(x_train)[:, 1])
    val_logloss = log_loss(y_val, gbm.predict_proba(x_val)[:, 1])
    print('tr_logloss: ', tr_logloss)
    print('val_logloss: ', val_logloss)

    # 模型预测
    # n行k列的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率, 这里的1表示点击的概率
    y_pred = gbm.predict_proba(test)[:, 1]
    # 这里看前10个， 预测为点击的概率
    print('predict: ', y_pred[:10])


if __name__ == '__main__':
    data0 = reader.load()
    print('type = ', type(data0))
    print('data0.shape = ', data0.shape)
    # 模型训练和预测GBDT模型
    gbdt_model(data0)


