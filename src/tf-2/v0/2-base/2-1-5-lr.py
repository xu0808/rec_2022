# -*- coding:utf-8 -*-
# tf自定义线下层

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# 1、完全自定义参数
class Linear_1(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear_1, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype='float32'), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype='float32'), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# x = tf.ones((2, 2))
# linear_layer = Linear_1(4, 2)
# y = linear_layer(x)
# print('Linear_1 = ', y)

# 2、add_weight定义参数
class Linear_2(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear_2, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# x = tf.ones((2, 2))
# linear_layer = Linear_2(4, 2)
# y = linear_layer(x)
# print('Linear_2 = ', y)


# 3、add_weight定义参数并动态感知特征维度
class Linear_3(layers.Layer):
    def __init__(self, units=32):
        super(Linear_3, self).__init__()
        self.units = units
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# x = tf.ones((2, 2))
# linear_layer = Linear_3(4, 2)
# y = linear_layer(x)
# print('Linear_3 = ', y)


# 4、自定义层建模
inputs = tf.keras.Input(shape=(32,))
x = Linear_3(units=64)(inputs)  # 等价于Dense()
x = tf.nn.relu(x)
x = Linear_3(units=64)(x)
x = tf.nn.relu(x)
y = layers.Dense(10)(x)

model = tf.keras.Model(inputs=inputs, outputs=y)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model.fit(data, labels, batch_size=32, epochs=5)
