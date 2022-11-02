# -*- coding:utf-8 -*-
# tf三种建模方式
# 构建模型（顺序模型、函数式模型、子类模型）
# 模型训练：model.fit()
# 模型验证：model.evaluate()
# 模型预测：model.predict()

import tensorflow as tf
print(tf.__version__)

image_input = tf.keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = tf.keras.Input(shape=(20, 10), name='ts_input')

x1 = tf.keras.layers.Conv2D(3, 3)(image_input)
x1 = tf.keras.layers.GlobalMaxPooling2D()(x1)

x2 = tf.keras.layers.Conv1D(3, 3)(timeseries_input)
x2 = tf.keras.layers.GlobalMaxPooling1D()(x2)

x = tf.keras.layers.concatenate([x1, x2])

score_output = tf.keras.layers.Dense(1, name='score_output')(x)
class_output = tf.keras.layers.Dense(5, name='class_output')(x)

model = tf.keras.Model(inputs=[image_input, timeseries_input],
                       outputs=[score_output, class_output])
# 打印模型结构
model.summary()

# 打印模型图问题解决
# https://blog.csdn.net/weixin_42459037/article/details/84066164

# 必须安装pydot和graphviz才能使pydotprint正常工作
# https://zhuanlan.zhihu.com/p/362085352
import os
os.environ["PATH"] += os.pathsep + 'C:/work/ide/Graphviz/bin/'
img = 'C:\\work\\workspace\\study\\rec\\tf-2\\2-base\\model.png'
tf.keras.utils.plot_model(model, img, show_shapes=True)


model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss=[tf.keras.losses.MeanSquaredError(),
          tf.keras.losses.CategoricalCrossentropy(from_logits=True)])

# Generate dummy Numpy data
import numpy as np
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# Fit on lists
model.fit([img_data, ts_data], [score_targets, class_targets], batch_size=32, epochs=3)

# Alternatively, fit on dicts
model.fit({'img_input': img_data, 'ts_input': ts_data},
          {'score_output': score_targets, 'class_output': class_targets}, batch_size=32, epochs=3)