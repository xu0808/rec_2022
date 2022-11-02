# -*- coding:utf-8 -*-
# 小 ResNet 模型
# https://tensorflow.google.cn/guide/keras/functional

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, add, GlobalAveragePooling2D, Dropout
# 查看版本
print('tf.version = ', tf.__version__)

data_dir = os.path.dirname('C:\\study\\data\\')
model_png = os.path.join(data_dir, 'temp', 'mini_resnet.png')


def model1():
    # [(None, 32, 32, 3)] ->  0
    inputs = keras.Input(shape=(32, 32, 3), name="img")
    # [(None, 30, 30, 32)] ->  896 (32 * (3*3*3 + 1))
    x = Conv2D(32, 3, activation="relu")(inputs)
    # [(None, 28, 28, 64)] ->  18496 (64 * (3*3*32 + 1))
    x = Conv2D(64, 3, activation="relu")(x)
    # (None, 9, 9, 64)
    block_1_output = MaxPooling2D(3)(x)

    x = Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = add([x, block_1_output])

    x = Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = add([x, block_2_output])

    x = Conv2D(64, 3, activation="relu")(block_3_output)
    # 输出(batch_size, channels)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10)(x)
    model = keras.Model(inputs, outputs, name="toy_resnet")

    # 打印模型结构
    model.summary()
    # 模型绘制为计算图
    keras.utils.plot_model(model, model_png, show_shapes=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )
    # We restrict the data to the first 1000 samples so as to limit execution time
    # on Colab. Try to train on the entire dataset until convergence!
    model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=10, validation_split=0.2)


if __name__ == '__main__':
    print('hello world!')
    model1()
