# -*- coding:utf-8 -*-
# 王喆深度推荐模型基础
# tf.feature_column特征预处理
# https://blog.csdn.net/BIT_666/article/details/119234194=

import os
import tensorflow as tf
from tensorflow.python.data.experimental import make_csv_dataset as csv_data
from tensorflow.keras.metrics import AUC

# 批大小设置
batch_size = 12
# 向量大小
emb_size = 10
# 模型训练迭代次数
epoch_num = 5
# 评价指标
metrics = ['accuracy', AUC(curve='ROC'), AUC(curve='PR')]

# 1、样本数据
# 王喆课程数据文件
data_dir = 'C:\\work\\data\\rec\\wangze\\sampledata'
# 训练样本
training_samples = os.path.join(data_dir, 'trainingSamples.csv')
# 测试样本
test_samples = os.path.join(data_dir, 'testSamples.csv')


# 数据文件读取
# 注意数据格式:每批一个集合，包含特征和标签，所有都是批的集合
def get_dataset(file_path):
    dataset = csv_data(file_path, batch_size=batch_size, label_name='label', na_value="0",
                       num_epochs=1, ignore_errors=True)
    return dataset


# 训练数据集
train_dataset = get_dataset(training_samples)
# 测试数据集
test_dataset = get_dataset(test_samples)

# 2、特征描述
# 2-1、电影风格字典
genre_dict = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy',
               'Western', 'Documentary', 'Sci-Fi', 'Drama', 'Thriller', 'Crime', 'Fantasy',
               'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']
# 2-2、分类特征
genre_features = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5',
                  'movieGenre1', 'movieGenre2', 'movieGenre3']

# 2-3、数值特征
num_features = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
                'userRatingCount', 'userAvgRating', 'userRatingStddev']

# 3、输入定义
inputs = {}
for feature in num_features:
    dtype = 'int32' if feature.endswith('Count') or feature.endswith('Year') else 'float32'
    inputs[feature] = tf.keras.layers.Input(name=feature, shape=(), dtype=dtype)
for feature in genre_features:
    inputs[feature] = tf.keras.layers.Input(name=feature, shape=(), dtype='string')
for feature in ['movieId', 'userId'] + ['userRatedMovie%d' % i for i in [1, 2, 3, 4, 5]]:
    inputs[feature] = tf.keras.layers.Input(name=feature, shape=(), dtype='int32')


def cate_col(key):
    # 分类特征转换成了one-hot特征
    return tf.feature_column.categorical_column_with_vocabulary_list(key=key, vocabulary_list=genre_dict)


def id_col(key, num_buckets):
    # 把id转换成one-hot特征
    return tf.feature_column.categorical_column_with_identity(key=key, num_buckets=num_buckets)


def emb_cate_col(key):
    # 映射到向量
    return tf.feature_column.embedding_column(cate_col(key), emb_size)


def emb_id_col(key, num_buckets):
    # 映射到向量
    return tf.feature_column.embedding_column(id_col(key, num_buckets), emb_size)


def one_hot_cate_col(key):
    # 转换成one-hot特征
    return tf.feature_column.indicator_column(cate_col(key))


def one_hot_id_col(key, num_buckets):
    # 转换成one-hot特征
    return tf.feature_column.indicator_column(id_col(key, num_buckets))


def num_col(key):
    # 数据列
    return tf.feature_column.numeric_column(key=key, default_value=0)


# 特征加工
def feature():
    # 1、分类特征
    cate_cols = []
    # 分类特征转换成向量
    for col in genre_features:
        cate_cols.append(emb_cate_col(key=col))
    # 2、编号特征处理
    # 电影编号转换成向量
    movie_emb_col = emb_id_col(key='movieId', num_buckets=1001)
    # 用户编号转换成向量
    user_emb_col = emb_id_col(key='userId', num_buckets=30001)
    # 3、数值特征
    num_columns = [num_col(key=col) for col in num_features]
    # 4、所有特征
    return num_columns + [movie_emb_col, user_emb_col] + cate_cols


# 4、模型操作
def model_main(model):
    # 打印模型结构
    # model.summary()
    # 4-1、模型编译
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    # 4-2、模型训练
    model.fit(train_dataset, epochs=epoch_num)

    # 4-3、模型评估
    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
    print('Test Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'
          .format(test_loss, test_accuracy, test_roc_auc, test_pr_auc))
    # 4-4、模型预测
    predictions = model.predict(test_dataset)
    # 标签:第1批样本的第二个字段
    test_dataset_bach_0 = list(test_dataset)[0][1]
    for prediction, goodRating in zip(predictions[:batch_size], test_dataset_bach_0):
        print("Predicted good rating: {:.2%}".format(prediction[0]),
              " | Actual rating label: ", ("Good Rating" if bool(goodRating) else "Bad Rating"))
