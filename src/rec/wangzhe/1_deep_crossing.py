# -*- coding:utf-8 -*-
# Deep Crossing模型

import base
import tensorflow as tf

# 1、所有特征
features = base.feature()

# 2、模型定义
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(features),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# 3、模型训练
base.model_main(model)
