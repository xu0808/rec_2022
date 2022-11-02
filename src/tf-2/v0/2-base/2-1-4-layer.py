# -*- coding:utf-8 -*-
# tf常用层

import tensorflow as tf

# 1、文本分类
a = tf.random.uniform(shape=(10, 100, 50), minval=-0.5, maxval=0.5)  # 张量
x = tf.keras.layers.LSTM(100)(a)  # LSTM
x = tf.keras.layers.Dense(10)(x)  # 全连接层
x = tf.nn.softmax(x)  # 激活函数
tf.print('x = ', x)

# 2、层参数配置
# 层中增加激活函数
tf.keras.layers.Dense(64, activation='relu')
# or
tf.keras.layers.Dense(64, activation=tf.nn.relu)
# 将L1正则化系数为0.01的线性层应用于内核矩阵
tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
# 将L2正则化系数为0.01的线性层应用于偏差向量：
tf.keras.layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
# 内核初始化为随机正交矩阵的线性层：
tf.keras.layers.Dense(64, kernel_initializer='orthogonal')
# 偏差向量初始化为2.0的线性层：
tf.keras.layers.Dense(64, bias_initializer=tf.keras.initializers.Constant(2.0))
