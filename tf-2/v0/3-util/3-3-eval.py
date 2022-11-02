# -*- coding:utf-8 -*-
# 自定义评估函数

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
import os
import matplotlib.pyplot as plt

print(tf.__version__)
print(np.__version__)
data_dir = os.path.dirname('C:\\study\\data\\commom\\')
mnist_file = os.path.join(data_dir, 'mnist.npz')
mnist = np.load(mnist_file)
x_train, y_train, x_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

# 图片显示
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


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


# 返回的是一个正确的个数
class CatgoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_true_positives', **kwargs):
        super(CatgoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        self.true_positives.assign(0.)


for x, y in train_ds:
    a = x
    b = y
    break
tf.print('tf.shape(a) = ', tf.shape(a))
tf.print('tf.shape(b) = ', tf.shape(b))

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()  # 损失函数
optimizer = tf.keras.optimizers.Adam()  # 优化器
# 评估函数
train_loss = tf.keras.metrics.Mean(name='train_loss')  # loss
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')  # 准确率
train_tp = CatgoricalTruePositives(name="train_tp")  # 返回正确的个数
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_tp = CatgoricalTruePositives(name='test_tp')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 评估函数的结果
    train_loss(loss)
    train_accuracy(labels, predictions)
    train_tp(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_tp(labels, predictions)


EPOCHS = 5
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_tp.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    test_tp.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, TP: {},Test Loss: {}, Test Accuracy: {}, Test TP:{}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          train_tp.result(),
                          test_loss.result(),
                          test_accuracy.result() * 100,
                          test_tp.result()))
