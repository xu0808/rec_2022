# -*- coding:utf-8 -*-
# DeepFM模型
from base import emb_id_col, emb_cate_col, one_hot_id_col, one_hot_cate_col, num_col
from base import num_features, inputs, model_main
import tensorflow as tf


# 1、特征处理
# 电影编号
movie_emb = emb_id_col(key='movieId', num_buckets=1001)
movie_one_hot = one_hot_id_col(key='movieId', num_buckets=1001)
# 用户编号
user_emb = emb_id_col(key='userId', num_buckets=30001)
user_one_hot = one_hot_id_col(key='userId', num_buckets=30001)
# 用户风格
user_genre_emb = emb_cate_col(key="userGenre1")
user_genre_one_hot = one_hot_cate_col(key="userGenre1")
# 电影风格
item_genre_emb = emb_cate_col(key="movieGenre1")
item_genre_one_hot = one_hot_cate_col(key="movieGenre1")

# 2、模型定义
# 2-1、线性层
# 分类特征
cate_features = [movie_one_hot, user_one_hot, user_genre_one_hot, item_genre_one_hot]
line_layer = tf.keras.layers.DenseFeatures(cate_features)(inputs)

# 2-2、交叉层
# 向量输入层
emb_features = [movie_emb, user_emb, item_genre_emb, user_genre_emb]
emb_layers = [tf.keras.layers.DenseFeatures([emb])(inputs) for emb in emb_features]
# 交叉层([i_u, i_g_u_g, i_g_u, u_g_i])
cross_indexs = [[0, 1], [2, 3], [2, 1], [3, 0]]
cross_layers = [tf.keras.layers.Dot(axes=1)([emb_layers[i], emb_layers[j]]) for [i, j] in cross_indexs]

# 2-3、DNN层
dense_features = [num_col(key=col) for col in num_features] + [movie_emb, user_emb]
deep = tf.keras.layers.DenseFeatures(dense_features)(inputs)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)

# 2-4、各层拼接（线性层 + 交叉层 + 深度网络层）
all_layer = [line_layer] + cross_layers + [deep]
concat_layer = tf.keras.layers.concatenate(all_layer, axis=1)

# 激活函数输出
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)
model = tf.keras.Model(inputs, output_layer)

# 3、模型训练
model_main(model)

