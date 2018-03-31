#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 11:58
# @Author  : Shanshan Fu
# @File    : linear_regression_sgd.py  :
# @Contact : 33sharewithu@gmail.com

import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('../..', '31_scaler/codes')))

class LinearRegressionSGD:
    def __init__(self, alpha = 0.01, max_iter = 100):
        self.alpha = alpha
        self.max_iter = max_iter
        self.loss = []

    def fit(self, X, y):
        self.w = np.random.randn(X.shape[1]+1, 1)
        X = np.insert(X, 0, 1, axis=1) # 插入截距项1
        y = y.reshape(-1, 1) # reshape (-1, 1)将y强制转成一列矩阵
        for _ in range(self.max_iter):
            y_pred = np.dot(X, self.w)
            loss = np.mean(0.5 * (y_pred - y)**2)
            self.loss.append(loss)
            print('%d iterator, loss is %f' % (_, loss))
            grad_w = np.dot(X.T, (y_pred - y))
            self.w -= self.alpha * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.w)
