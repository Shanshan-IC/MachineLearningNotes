#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class SoftmaxRegression:
    def __init__(self, alpha = 0.01, max_iter = 100):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        self.num_class = len(set(y))
        self.w = np.random.randn(self.num_class, X.shape[1] + 1)
        X = np.insert(X, 0, 1, axis=1)  # 插入截距项1
        y = y.reshape(-1, 1)  # reshape (-1, 1)将y强制转成一列矩阵
        for _ in xrange(self.max_iter):
            for i in xrange(self.num_class):
                grad_w = self.cal_prob(X, y, i)
                self.w[i] -= self.alpha * grad_w.T.reshape(-1)

    def _dot(self, X, j):
        w_j = self.w[j]
        return np.exp(np.dot(X, w_j))

    def cal_prob(self, X, y, j):
        first = (y == j).astype(int)
        molecule = self._dot(X, j).reshape(-1, 1)
        denominator = np.sum(np.exp(np.dot(X, self.w.T)), axis = 1).reshape(-1, 1)
        prob = molecule / denominator
        return np.dot(X.T, (first - prob))

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # 插入截距项1
        y_pred = np.dot(X, self.w.T)
        row, col = y_pred.shape
        pos = np.argmax(y_pred, axis=1)
        m, n = divmod(pos, col)
        return m