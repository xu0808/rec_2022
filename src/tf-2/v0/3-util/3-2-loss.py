# -*- coding:utf-8 -*-
# 自定义损失函数

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os

print(tf.__version__)
print(np.__version__)
data_dir = os.path.dirname('C:\\study\\data\\commom\\')
mnist_file = os.path.join(data_dir, 'mnist.npz')
mnist = np.load(mnist_file)
x_train, y_train, x_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = np.int32(y_train)
y_test = np.int32(y_test)
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(100).batch(32)


def MyModel():
    inputs = tf.keras.Input(shape=(28, 28, 1), name='digits')
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def FocalLoss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

        y_true = tf.cast(y_true, tf.float32)

        loss = -  y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)

        loss = tf.math.reduce_sum(loss, axis=1)
        return loss

    return focal_loss_fixed


model = MyModel()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),  # 优化器
              loss=FocalLoss(gamma=2.0, alpha=0.25),  # 损失函数
              metrics=[tf.keras.metrics.CategoricalAccuracy()]
              )  # 评估函数
model.fit(train_ds, epochs=5, validation_data=test_ds)
