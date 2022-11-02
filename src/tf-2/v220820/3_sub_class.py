# -*- coding:utf-8 -*-
#  Keras 模型子类化
# https://tensorflow.google.cn/tutorials/quickstart/advanced

import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, concatenate
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 查看版本
print('tf.version = ', tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, keras, tf:
    print(module.__name__, module.__version__)

# 1.加载数据集 波士顿房价预测
housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)
# 2.拆分数据集
# 训练集与测试集拆分
data = train_test_split(housing.data, housing.target, random_state=7, test_size=0.20)
x_train_all, x_test, y_train_all, y_test = data
# 训练集与验证集的拆分
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11, test_size=0.20)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)
# 3、数据预处理 数据集的归一化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)


# 4、网络模型的搭建
# 子类API
class WideDeepModel(tf.keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        self.hidden1_layer = Dense(30, activation='relu')
        self.hidden2_layer = Dense(30, activation='relu')
        self.output_layer = Dense(1)

    def call(self, inputs, training=None, mask=None):
        """完成模型的正向计算"""
        hidden1 = self.hidden1_layer(inputs)
        hidden2 = self.hidden2_layer(hidden1)
        concat = concatenate([inputs, hidden2])
        output = self.output_layer(concat)
        return output


model = WideDeepModel()
model.build(input_shape=(None, 8))
print(model.layers)
model.summary()

# 5、模型的编译  设置损失函数 优化器
model.compile(loss='mean_squared_error', optimizer='adam')

# 6、设置回调函数
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

# 7、训练网络
history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid), epochs=10, callbacks=callbacks)

# 8、绘制训练过程数据
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

if __name__ == '__main__':
    print('hello world!')