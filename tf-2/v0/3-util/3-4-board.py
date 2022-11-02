# -*- coding:utf-8 -*-
# Tensorboard界面解释
# tensorboard --logdir .\tf-2\3-util\keras_logv2
# Scalars : 显示了如何将loss与每个时间段改变。还可以使用它来跟踪训练速度，
# 学习率和其他标量值。
# Graphs：进行可视化模型。在这种情况下，将显示层的Keras图，这可以帮助你
# 确保模型正确构建。
# Distributions 和 Histograms ：显示张量随时间的分布。这对于可视化权重
# 和偏差并验证它们是否以预期的方式变化很有用。

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
import os

print(tf.__version__)
print(np.__version__)
data_dir = os.path.dirname('C:\\study\\data\\commom\\')
mnist_file = os.path.join(data_dir, 'mnist.npz')
mnist = np.load(mnist_file)
x_train, y_train, x_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


model = MyModel()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="keras_logv2",
                                                      histogram_freq=1,
                                                      profile_batch=100000000)
model.fit(x=x_train, y=y_train, epochs=20,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])
