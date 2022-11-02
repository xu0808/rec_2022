# -*- coding:utf-8 -*-
# 自定义线性回归

import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target
print('data.shape = ', data.shape)
print('target.shape = ', target.shape)


# 自定义全连接层
class Linear1(tf.keras.layers.Layer):

    def __init__(self, units=1, input_dim=4):
        super(Linear1, self).__init__()  #
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype='float32'), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class Linear2(tf.keras.layers.Layer):

    def __init__(self, units=1, input_dim=4):
        super(Linear2, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class Linear3(tf.keras.layers.Layer):

    def __init__(self, units=32):
        super(Linear3, self).__init__()
        self.w = None
        self.b = None
        self.units = units

    def build(self, input_shape):  # (150,4)
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(Linear3, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


tf.keras.backend.set_floatx('float64')
x = tf.constant(data)  # (150,4)
# linear_layer = Linear2(units=1, input_dim=4)
linear_layer = Linear3(units=1)
y = linear_layer(x)
print(y.shape)
