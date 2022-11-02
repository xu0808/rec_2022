# -*- coding:utf-8 -*-
# 张量定义

import tensorflow as tf

# 张量定义
# Rank0
mammal = tf.Variable("Elephant", tf.string)
tf.print(tf.rank(mammal))
tf.print(tf.shape(mammal))
print(tf.rank(mammal))

# Rank1
mystr = tf.Variable(["Hello"], tf.string)
tf.print(tf.rank(mystr))
tf.print(tf.shape(mystr))

# Rank2
mymat = tf.Variable([[7], [11]], tf.int16)
tf.print(tf.rank(mymat))
tf.print(tf.shape(mymat))

tf.print(tf.constant([1, 2, 3], dtype=tf.int16))

tf.print(tf.zeros((2, 2), dtype=tf.int16))

# reshape
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])
tf.print(matrix)
# 普通数值
print(rank_three_tensor.numpy())
