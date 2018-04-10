#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/15 17:01

# 方法1：求解逆矩阵
import numpy as np
class LinearRegression:
    def fit(self, X, y):
        A = np.dot(X.T, X)
        self.theta = np.dot(np.dot(np.linalg.inv(A), X.T), y) # 看readme公式

    def predict(self, X):
        return np.dot(X, self.theta)
