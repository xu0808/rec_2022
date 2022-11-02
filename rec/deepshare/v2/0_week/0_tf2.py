#!/usr/bin/env.txt python
# coding: utf-8
# Tensorflow 张量与常用函数
# https://www.cnblogs.com/mllt/p/tensorflow01.html

# tf版本查看
import tensorflow as tf
import numpy as np

# 打印版本和 2.3.0
print(tf.__version__)
print(tf.test.is_gpu_available())


# gpu = tf.config.list_physical_devices('GPU')
# print(gpu)

def ini_tensor():
    """创建张量"""
    # 创建一个最简单的张量
    a = tf.constant([1, 5], dtype=tf.int64)
    print('a = ', a)
    print('a.dtype = ', a.dtype)
    print('a.shape = ', a.shape)
    print('a.numpy() = ', a.numpy())

    # numpy转为tensor
    b = np.arange(0, 5)
    c = tf.convert_to_tensor(b, dtype=tf.int64)
    print('b.dtype = ', b.dtype)
    print('b = ', b)
    print('c.dtype = ', c.dtype)
    print('c.numpy() = ', c.numpy())
    print('c = ', c)

    # 通过维度创建一个tensor
    zeros = tf.zeros([2, 3])
    print('zeros.dtype = ', zeros.dtype)
    print('zeros.shape = ', zeros.shape)
    print('zeros.numpy() = ', zeros.numpy())
    print('zeros = ', zeros)
    ones = tf.ones(4)
    print('ones.dtype = ', ones.dtype)
    print('ones.shape = ', ones.shape)
    print('ones.numpy() = ', ones.numpy())
    print('ones = ', ones)
    fill = tf.fill([2, 2], 9)
    print('fill.dtype = ', fill.dtype)
    print('fill.shape = ', fill.shape)
    print('fill.numpy() = ', fill.numpy())

    # 生成正态分布的随机数(默认均值为0，标准差为1)
    d = tf.random.normal([2, 3], mean=0.5, stddev=1)  # stddev 标准差
    print('d = ', d)
    # 生成截断式正态分布的随机数（生成的随机数更集中一些）
    e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
    print('e = ', e)
    # 生成指定维度的均匀分布的随机数[minval,maxval)
    f = tf.random.uniform([2, 3], minval=0, maxval=1)
    print('f = ', f)


def base_fun():
    """常用函数"""
    # 强制tensor转换为该数据类型#
    x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
    print(x1)
    x2 = tf.cast(x1, tf.int32)
    print(x2)

    # 计算张量维度上元素的最大、小值
    print(tf.reduce_min(x2), '\n', tf.reduce_max(x2))
    # axis：用于指定操作方向
    # 计算张量沿着指定维度的平均值
    # 计算张量沿着指定维度的和
    x = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(x)
    # 计算张量沿着指定维度的平均值 所有元素参与计算
    print(tf.reduce_mean(x))
    # 计算张量沿着指定维度的和  跨列操作
    print('sum0 = ', tf.reduce_sum(x, axis=0))
    print('sum1 = ', tf.reduce_sum(x, axis=1))

    # 增加张量的维度
    t = tf.constant([1, 2])
    print('t = ', t)
    # 在第1个维度加1维（就是在第1维外加[]）
    t1 = tf.expand_dims(t, 0)
    print('t1 = ', t1)
    # 在第2个维度加1维（就是在第2维外加[]）
    t2 = tf.expand_dims(t, 1)
    print('t2 = ', t2)
    # 在最后1个维度加1维（就是在最后1维外加[]）
    t3 = tf.expand_dims(t, -1)
    print('t3 = ', t3)

    # 将变量标记为“可训练”
    w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
    print(w)


def math_fun():
    """数学运算"""
    # 对应元素的四则运算（加减乘除:
    # tf.add，tf.subtract，tf.multiply，tf.divide
    # 矩阵乘：
    # tf.matmul
    # 平方：tf.square、次方：tf.pow、开方：tf.sqrt
    a = tf.ones([1, 3])
    b = tf.fill([1, 3], 3.)
    print(a)
    print(b)
    print(tf.add(a, b))
    print(tf.subtract(a, b))
    print(tf.multiply(a, b))
    print(tf.divide(b, a))
    print(tf.pow(a, 3))  # 三次方
    print(tf.square(a))  # 平方
    print(tf.sqrt(a))  # 开方

    a = tf.ones([3, 2])
    b = tf.fill([2, 3], 3.)
    print(a)
    print(b)
    print(tf.matmul(a, b))


def gradient():
    """求导运算"""
    with tf.GradientTape() as tape:
        w = tf.Variable(tf.constant(3.0))  # 标记为“可训练”
        loss = tf.pow(w, 2)  # 求w的2次方
    # 损失函数loss 对 参数w 的求导数运算
    grad = tape.gradient(loss, w)
    print(grad)


if __name__ == '__main__':
    print('tf2.0 tensor')
    # ini_tensor()
    base_fun()
    # math_fun()
    # gradient()
