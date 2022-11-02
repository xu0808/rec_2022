# -*- coding:utf-8 -*-
# DIN模型
from base import emb_id_col, emb_cate_col, num_col
from base import num_features, inputs, model_main
import tensorflow as tf

# userRatedMovie{1-5}
recent_movies = 5

# 1、特征处理
# 候选集特征
movie_emb = emb_id_col(key='movieId', num_buckets=1001)
# 行为特征
recent_rate = [num_col(key='userRatedMovie%d' % i) for i in [1, 2, 3, 4, 5]]
# 属性特征
user_emb = emb_id_col(key='userId', num_buckets=30001)
user_genre_emb = emb_cate_col(key="userGenre1")
user_profile = [user_emb, user_genre_emb] + [num_col(key=f) for f in num_features[4:]]
# 场景特征
item_genre_emb = emb_cate_col(key="movieGenre1")
context_features = [item_genre_emb] + [num_col(key=f) for f in num_features[:4]]

# 2、模型定义
# 2-1、向量输入层[候选集、行为、属性、场景]
emb_features = [movie_emb, recent_rate, user_profile, context_features]
emb_layers = [tf.keras.layers.DenseFeatures([emb])(inputs) for emb in emb_features]

# 2-2、激活层
# 候选集样本放大
candidate_layer = tf.keras.layers.RepeatVector(recent_movies)(emb_layers[0])
# 行为激活层
behaviors_layer = tf.keras.layers.Embedding(input_dim=1001, output_dim=10, mask_zero=True)(emb_layers[1])
# 激活层相减
activation_sub_layer = tf.keras.layers.Subtract()([behaviors_layer, candidate_layer])
# 激活层相乘
activation_product_layer = tf.keras.layers.Multiply()([behaviors_layer, candidate_layer])
# 激活层拼接
layers = [activation_sub_layer, behaviors_layer, candidate_layer, activation_product_layer]
activation_all = tf.keras.layers.concatenate(layers, axis=-1)

# 激活层DNN
activation_unit = tf.keras.layers.Dense(32)(activation_all)
activation_unit = tf.keras.layers.PReLU()(activation_unit)
activation_unit = tf.keras.layers.Dense(1, activation='sigmoid')(activation_unit)
activation_unit = tf.keras.layers.Flatten()(activation_unit)
activation_unit = tf.keras.layers.RepeatVector(10)(activation_unit)
activation_unit = tf.keras.layers.Permute((2, 1))(activation_unit)
# 行为层和DNN输出层相乘
activation_unit = tf.keras.layers.Multiply()([behaviors_layer, activation_unit])
# 最大池化
user_behaviors_pooled_layers = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(activation_unit)

# 2-2、全连接层
concat_layers = [emb_layers[2], user_behaviors_pooled_layers, candidate_layer, emb_layers[3]]
concat_layer = tf.keras.layers.concatenate(concat_layers)
output_layer = tf.keras.layers.Dense(128)(concat_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(64)(output_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
# 激活函数
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)
model = tf.keras.Model(inputs, output_layer)
model.summary()

# 3、模型训练
model_main(model)