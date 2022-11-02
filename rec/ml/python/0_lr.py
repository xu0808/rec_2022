#!/usr/bin/env.txt python
# coding: utf-8
# python手动实现线性回归

# 手撕机器学习01-线性回归
# https://blog.csdn.net/blowfire123/article/details/109995593
"""
梯度下降本质:
x = x - y'(x) * α
"""

import numpy as np

# 1、生成数据集
# 样本数量、特征维度
num, dim = 10000, 2
w_0 = [4, -2.4]  # 设置真实w = [4, -2.4]⊤和偏差b = 3.2
b_0 = 3.2
features = np.random.normal(scale=1, size=(num, dim))
labels = w_0[0] * features[:, 0] + w_0[1] * features[:, 1] + b_0
# 增加噪音
labels += np.random.normal(scale=0.01, loc=0, size=labels.shape)
print('features.shape = ', features.shape, 'labels.shape = ', labels.shape)

# 批大小
batch_size = 100
indices = np.array(range(num))
# 随机打乱顺序
np.random.shuffle(indices)
x_0 = np.array([features.take(indices[i:i + batch_size], 0) for i in range(0, num, batch_size)])
y_0 = np.array([labels.take(indices[i:i + batch_size], 0) for i in range(0, num, batch_size)])
print('x_0.shape = ', x_0.shape, 'y_0.shape = ', y_0.shape)


# 2、定义模型
class LinearRegression:
    def __init__(self, x_dim):
        # 初始化参数
        self.w = np.random.normal(scale=0.01, size=(x_dim, 1))
        self.b = np.random.normal(scale=0.01, size=(1,))

    def __str__(self):
        return 'w = ' + str(self.w.reshape((dim, ))) + '\nb = ' + str(self.b[0])

    def forward(self, x):
        # 前向计算
        return np.dot(x, self.w) + self.b

    def mse(self, pre, y):
        # 均方误差计算
        return (pre - y.reshape(pre.shape)) ** 2 / 2

    def sgd(self, learning_rate, x, y_hat, y):
        # batch梯度下降，反向传播
        # 外层偏导(不含pre的一阶偏导)
        grad = (y_hat - y) / len(y)
        self.w -= learning_rate * np.dot(x.T, grad)
        self.b -= learning_rate * sum(grad)

    def train(self, learning_rate, x, y, epoch):
        for i in range(epoch):
            loss = []
            for input, label in zip(x, y):
                output = self.forward(input)
                label = label.reshape(output.shape)
                loss.append(self.mse(output, label))
                self.sgd(learning_rate, input, output, label)
            print('epoch %d, loss %f' % (i + 1, np.mean(loss)))


# 3、迭代训练
if __name__ == '__main__':
    model = LinearRegression(dim)
    print(model)
    model.train(0.0012, x_0, y_0, 100)
    print(model)
    print('w_0 = ', w_0, '\nb_0 = ', b_0)
