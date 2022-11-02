# -*- coding:utf-8 -*-
# tf基本操作

import tensorflow as tf

# 1、tf.strings
# 字符切割
tf.print(tf.strings.bytes_split('hello'))
print('tf.strings.bytes_split(\'hello\') = ', tf.strings.bytes_split('hello'))

# 单词切割
tf.print(tf.strings.split('hello world'))

# string hash
tf.print(tf.strings.to_hash_bucket(['hello', 'world'], num_buckets=10))

# 2、tf.debugging
# tf自带debug函数
a = tf.random.uniform((10, 10))
tf.debugging.assert_equal(x=a.shape, y=(10, 10))
# # 错误示范
# tf.debugging.assert_equal(x=a.shape, y=(20, 10))

# 3、tf.random
a = tf.random.uniform(shape=(10, 5), minval=0, maxval=10)
tf.print(a)
print('a = ', a)

# 4、tf.math
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

tf.print(tf.math.add(a, b))
tf.print(tf.math.subtract(a, b))
tf.print(tf.math.multiply(a, b))
tf.print(tf.math.divide(a, b))
print(tf.math.add(a, b))

# 5、tf.dtypes
x = tf.constant([1.8, 2.2], dtype=tf.float32)
x1 = tf.dtypes.cast(x, tf.int32)
print('x1=', x1)
