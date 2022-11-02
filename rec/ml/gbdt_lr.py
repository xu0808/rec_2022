# -*- coding:utf-8 -*-
# GBDT+LR
# https://blog.csdn.net/weixin_42691585/article/details/109337381
# 3. LR+GBDT模型： 由于LR使用的特征是GBDT的输出， 原数据依然是GBDT进行处理交叉， 所以只需要离散特征one-hot处理

import reader
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
import gc
import warnings
warnings.filterwarnings('ignore')


# LR + GBDT建模
# 下面就是把上面两个模型进行组合， GBDT负责对各个特征进行交叉和组合， 把原始特征向量转换为新的离散型特征向量， 然后在使用逻辑回归模型
def gbdt_lr_model(data):  # 0.43616
    # 离散特征one-hot编码
    for col in reader.category_cols:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    gbm = lgb.LGBMClassifier(objective='binary',
                             subsample=0.8,
                             min_child_weight=0.5,
                             colsample_bytree=0.7,
                             num_leaves=100,
                             max_depth=12,
                             learning_rate=0.01,
                             n_estimators=1000)

    gbm.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_names=['train', 'val'],
            eval_metric='binary_logloss',
            early_stopping_rounds=100)

    model = gbm.booster_

    gbdt_feats_train = model.predict(train, pred_leaf=True)
    gbdt_feats_test = model.predict(test, pred_leaf=True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

    train = pd.concat([train, df_train_gbdt_feats], axis=1)
    test = pd.concat([test, df_test_gbdt_feats], axis=1)
    train_len = train.shape[0]
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()

    # 连续特征归一化
    scaler = MinMaxScaler()
    for col in reader.continuous_cols:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    train = data[: train_len]
    test = data[train_len:]
    del data
    gc.collect()

    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3, random_state=2018)

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('val-logloss: ', val_logloss)

    y_pred = lr.predict_proba(test)[:, 1]
    # 这里看前10个， 预测为点击的概率
    print(y_pred[:10])


if __name__ == '__main__':
    data0 = reader.load()
    print('type = ', type(data0))
    print('data0.shape = ', data0.shape)
    # 训练和预测lr模型
    gbdt_lr_model(data0)
