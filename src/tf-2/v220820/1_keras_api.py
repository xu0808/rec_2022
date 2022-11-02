# -*- coding:utf-8 -*-
# 函数式 API
# https://tensorflow.google.cn/guide/keras/functional

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
# 查看版本
print('tf.version = ', tf.__version__)

data_dir = os.path.dirname('C:\\study\\data\\')
model_png = os.path.join(data_dir, 'temp', 'first_model.png')


def model1():
    # 定义3层神经网络
    inputs = keras.Input(shape=(784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
    # 打印模型结构
    model.summary()
    # 模型绘制为计算图
    keras.utils.plot_model(model, model_png, show_shapes=True)
    # 验证隐藏层网络参数
    print('hidden layer 1 weight', 784 * 64 + 64)
    print('hidden layer 2 weight', 64 * 64 + 64)
    print('hidden layer 3 weight', 64 * 10 + 10)


if __name__ == '__main__':
    print('hello world!')
    model1()
