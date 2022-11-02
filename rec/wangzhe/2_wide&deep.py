# -*- coding:utf-8 -*-
# Wide&Deep模型

import base
import tensorflow as tf

# 1、所有特征
features = base.feature()

# 2、交叉特征
movie_col = base.id_col(key='movieId', num_buckets=1001)
rated_movie = base.id_col(key='userRatedMovie1', num_buckets=1001)
# 特征交叉
crossed_col = tf.feature_column.crossed_column([movie_col, rated_movie], 10000)
# 交叉特征进行onehot变换
crossed_feature = tf.feature_column.indicator_column(crossed_col)


# 3、模型
# 3-1、模型定义
# deep part for all features
deep = tf.keras.layers.DenseFeatures(features)(base.inputs)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
# wide part for cross feature
wide = tf.keras.layers.DenseFeatures(crossed_feature)(base.inputs)
both = tf.keras.layers.concatenate([deep, wide])
output = tf.keras.layers.Dense(1, activation='sigmoid')(both)
model = tf.keras.Model(base.inputs, output)

# 3-2、模型训练
base.model_main(model)