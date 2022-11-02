#!/usr/bin/env.txt python
# coding: utf-8
# python手动实现逻辑回归
# 因为进行了非线性转化，所以并不能完全求解线性方程且精度和梯度有关

# 手撕机器学习02-逻辑回归
# https://blog.csdn.net/blowfire123/article/details/110114336

import numpy as np
from IPython import display
from matplotlib import pyplot as plt

# 1、生成数据集
num, dim = 10000, 2
w_0 = [4, -2.4]  # 设置真实w = [4, -2.4]⊤和偏差b = 3.2
b_0 = 3.2
features = np.random.normal(scale=5, size=(num, dim))
labels = w_0[0] * features[:, 0] + w_0[1] * features[:, 1] + b_0
labels = np.array(list(map(lambda x: 1 if x > 0 else 0, labels)))

# 批大小
batch_size = 100
indices = np.array(range(num))
# 随机打乱顺序
np.random.shuffle(indices)
x_0 = np.array([features.take(indices[i:i + batch_size], 0) for i in range(0, num, batch_size)])
y_0 = np.array([labels.take(indices[i:i + batch_size], 0) for i in range(0, num, batch_size)])
print('x_0.shape = ', x_0.shape, 'y_0.shape = ', y_0.shape)


def set_figsize(figsize=(5, 4)):
    # 用矢量图显示
    display.set_matplotlib_formats('svg')
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def show():
    set_figsize()
    colors = ['g', 'r', 'b']
    label_com = ['0', '1', 'line']
    x1_0, x2_0, x1_1, x2_1 = [], [], [], []
    for i in range(len(labels)):
        if labels[i] == 0:
            x1_0.append(features[i, 0])
            x2_0.append(features[i, 1])
        else:
            x1_1.append(features[i, 0])
            x2_1.append(features[i, 1])
    # y = w*x + b 的分割线
    line_x1 = np.arange(-10, 10, 0.1)
    line_x2 = (w_0[0] * line_x1 + b_0) / -w_0[1]
    plt.scatter(x1_0, x2_0, c=colors[0], cmap='brg', s=5, marker='8', linewidth=0)
    plt.scatter(x1_1, x2_1, c=colors[1], cmap='brg', s=5, marker='8', linewidth=0)
    plt.scatter(line_x1, line_x2, c=colors[2], cmap='brg', s=3, marker='8', linewidth=0)
    plt.legend(labels=label_com, loc='upper right')
    plt.show()


# 2、定义模型
class LogisticsRegression:
    def __init__(self, x_dim):
        # 初始化参数
        self.w = np.random.normal(scale=1, size=(x_dim, 1))
        self.b = np.random.normal(scale=0.01, size=(1,))

    def __str__(self):
        # 输出参数
        return 'w = ' + str(self.w.reshape((dim, ))) + '\nb = ' + str(self.b[0])

    def forward(self, x):
        # 前向计算
        return np.dot(x, self.w) + self.b

    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))

    def cross_entropy(self, y_hat, y):
        # 损失函数-交叉熵函数
        # y -> {0, 1}
        # l = -sum(y*ln(sigmoid(y_hat)) + (1-y)*ln(sigmoid(1-y_hat)))
        return - np.dot(y.T, np.log(y_hat)) - np.dot((1 - y).T, np.log(1.0000000001 - y_hat))

    def accuracy(self, y_hat, y):
        # 计算分类的准确率
        pre = (y_hat > 0.5)
        pre = np.array(list(map(lambda x: 1 if x else 0, pre))).reshape(y.shape)
        return np.sum(pre == y) / len(y)

    def sgd(self, learning_rate, x, y_hat, y):
        # batch梯度下降，反向传播
        # 外层偏导(不含pre的一阶偏导)
        grad_0 = (y_hat - y)/len(y)
        grad = np.dot(x.T, grad_0)
        self.w -= learning_rate * grad
        self.b -= learning_rate * sum(grad_0)

    def train(self, learning_rate, x, y, epoch):
        for i in range(epoch):
            loss, acc = [], []
            for input, label in zip(x, y):
                # wx + b
                output = self.forward(input)
                # sigmoid(wx + b)
                y_hat = self.sigmoid(output)
                label = label.reshape(output.shape)
                loss.append(self.cross_entropy(y_hat, label))
                self.sgd(learning_rate, input, y_hat, label)
                acc.append(self.accuracy(y_hat, label))
            print('epoch %d, loss %f, accuracy %f' % (i+1, np.mean(loss), np.mean(acc)))


if __name__ == "__main__":
    show()
    # model = LogisticsRegression(dim)
    # print(model)
    # model.train(0.018, x_0, y_0, 200)
    # print(model)
    # print('w_0 = ', w_0, '\nb_0 = ', b_0)
