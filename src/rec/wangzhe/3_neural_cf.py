# -*- coding:utf-8 -*-
# 双塔模型

import base
import tensorflow as tf

# 1、特征加工
# 电影编号
movie_emb_col = base.emb_id_col(key='movieId', num_buckets=1001)
# 用户编号
user_emb_col = base.emb_id_col(key='userId', num_buckets=30001)


# 2、双塔模型
# 2-1、网络一：拼接再卷积
def neural_cf_model_1(feature_inputs, item_feature, user_feature, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature)(feature_inputs)
    user_tower = tf.keras.layers.DenseFeatures(user_feature)(feature_inputs)
    interact_layer = tf.keras.layers.concatenate([item_tower, user_tower])
    for num_nodes in hidden_units:
        interact_layer = tf.keras.layers.Dense(num_nodes, activation='relu')(interact_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(interact_layer)
    neural_cf_model = tf.keras.Model(feature_inputs, output_layer)
    return neural_cf_model


# 2-2、网络二：卷积再点乘
def neural_cf_model_2(feature_inputs, item_feature, user_feature, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature)(feature_inputs)
    for num_nodes in hidden_units:
        item_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(item_tower)

    user_tower = tf.keras.layers.DenseFeatures(user_feature)(feature_inputs)
    for num_nodes in hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(user_tower)

    output = tf.keras.layers.Dot(axes=1)([item_tower, user_tower])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    neural_cf_model = tf.keras.Model(feature_inputs, output)
    return neural_cf_model


# 2-3、选择模型网络
model = neural_cf_model_1(base.inputs, [movie_emb_col], [user_emb_col], [10, 10])

# 3、模型训练
base.model_main(model)
