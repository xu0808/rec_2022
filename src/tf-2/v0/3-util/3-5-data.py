# -*- coding:utf-8 -*-
# dataset

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
import os

data_dir = os.path.dirname('C:\\study\\data\\commom\\')
mnist_file = os.path.join(data_dir, 'mnist.npz')
mnist = np.load(mnist_file)


# Dense
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        self.units = units
        super(MyDense, self).__init__(**kwargs)

    # build方法一般定义Layer需要被训练的参数。
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='w')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='b')
        super(MyDense, self).build(input_shape)  # 相当于设置self.built = True

    # call方法一般定义正向传播运算逻辑，__call__方法调用了它。
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
    def get_config(self):
        config = super(MyDense, self).get_config()
        config.update({'units': self.units})
        return config


from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
labels = iris.target

# 网络   函数式构建的网络
inputs = tf.keras.Input(shape=(4,))
x = MyDense(units=16)(inputs)
x = tf.nn.tanh(x)
x = MyDense(units=3)(x)  # 0,1,2
predictions = tf.nn.softmax(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

data = np.concatenate((data, labels.reshape(150, 1)), axis=-1)
np.random.shuffle(data)
labels = data[:, -1]
data = data[:, :4]

# 优化器 Adam
# 损失函数 交叉熵损失函数
# 评估函数 acc


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# keras
model.fit(data, labels, batch_size=32, epochs=100, shuffle=True)
