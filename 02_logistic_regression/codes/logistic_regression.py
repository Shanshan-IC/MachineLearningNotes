#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/30 13:47
# @Author  : Shanshan Fu
# @File    : logistic_regression.py  :
# @Contact : 33sharewithu@gmail.com
import numpy as np

class LogisticRegression:
    def __init__(self, alpha = 0.01, max_iter = 100):
        self.alpha = alpha
        self.max_iter = max_iter

    def sigmoid(self, y):
        return 1.0 / (1 + np.exp(-y))

    def fit(self, X, y):
        self.w = np.random.randn(X.shape[1] + 1, 1)
        X = np.insert(X, 0, 1, axis=1)  # 插入截距项1
        y = y.reshape(-1, 1)  # reshape (-1, 1)将y强制转成一列矩阵
        for _ in xrange(self.max_iter):
            y_pred = self.sigmoid(np.dot(X, self.w))
            grad_w = np.dot(X.T, (y_pred - y))
            self.w += self.alpha * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self.sigmoid(np.dot(X, self.w))