# -*- coding:utf-8 -*-
# tf版本查看
import tensorflow as tf

# 打印版本和 2.3.0
print(tf.__version__)
print(tf.test.is_gpu_available())

gpu = tf.config.list_physical_devices('GPU')
print(gpu)


