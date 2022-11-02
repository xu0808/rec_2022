# -*- coding:utf-8 -*-
# tf三种建模方式

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# 1、序列模型
# 第一种Sequential model
model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))  # 第一层
model.add(layers.Dense(64, activation='relu'))  # 第二层
model.add(layers.Dense(10))  # 第三层

# 第二种Sequential model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),  # 第一层
    layers.Dense(64, activation='relu'),  # 第二层
    layers.Dense(10)  # 第三层
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)

# 2、函数模型
# 函数式模型可以处理具有非线性拓扑的模型，具有共享层的模型以及具有多个输入或输出的模型，更灵活
inputs1 = tf.keras.Input(shape=(32,))  # 输入1
inputs2 = tf.keras.Input(shape=(32,))  # 输入2
x1 = layers.Dense(64, activation='relu')(inputs1)  # 第一层
x2 = layers.Dense(64, activation='relu')(inputs2)  # 第一层
x = tf.concat([x1, x2], axis=-1)
x = layers.Dense(64, activation='relu')(x)  # 第二层
predictions = layers.Dense(10)(x)  # 第三层
model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

data1 = np.random.random((1000, 32))
data2 = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model.fit((data1, data2), labels, batch_size=32, epochs=5)


# 3、子类模型
# 通过子类化tf.keras.Model和定义自己的前向传播模型来构建完全可定制的模型。
# 和eager execution模式相辅相成。
class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 定义自己需要的层
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes)

    def call(self, inputs):
        # 定义前向传播
        # 使用在 (in `__init__`)定义的层
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x


model = MyModel(num_classes=10)

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
