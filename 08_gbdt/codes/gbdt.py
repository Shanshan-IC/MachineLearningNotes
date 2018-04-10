#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score
import sys, os
sys.path.append(os.path.abspath(os.path.join('../..', '06_decision_tree/codes')))
from decision_tree_regression import *

class GBDT(object):
    def __init__(self, min_sample_splits, min_impurity, max_depth,
                 num_trees, learning_rate, regression):
        self.min_sample_splits = min_sample_splits
        self.min_impurity = min_impurity
        self.max_depth = max_depth

        self.num_trees = num_trees
        self.regression = regression
        self.learning_rate = learning_rate
        self.trees = []
        self.loss = SquareLoss()
        if not self.regression:
            self.loss = CrossEntropy()

        for _ in xrange(self.num_trees):
            tree = DecisionTreeRegression(min_sample_splits=self.min_sample_splits,
                                          min_impurity=self.min_impurity, max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X, y):
        # 初始化
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for tree in self.trees:
            gradient = self.loss.gradient(y, y_pred)
            tree.fit(X, gradient)
            update = tree.predict(X)
            y_pred -= np.multiply(self.learning_rate, update)

    def predict(self, X):
        y_pred = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if y_pred.any() else y_pred - update

        if not self.regression:
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred

class GBDTRegression(GBDT):
    def __init__(self, min_sample_splits, min_impurity, max_depth,
                 num_trees, learning_rate):
        super(GBDTRegression, self).__init__(min_sample_splits=min_sample_splits,
                                             min_impurity=min_impurity,
                                             max_depth=max_depth,
                                             num_trees=num_trees,
                                             learning_rate=learning_rate,
                                             regression=True)

class Loss(object):
    def loss(self, y, y_pred):
        return NotImplementedError

    def gradient(self, y, y_pred):
        return NotImplementedError

    def acc(self, y, y_pred):
        return 0

# 回归模型
class SquareLoss(Loss):
    def loss(self, y, y_pred):
        return 0.5 * np.power(y - y_pred, 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

# 分类模型
class CrossEntropy(Loss):
    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1-y) / (1-p)