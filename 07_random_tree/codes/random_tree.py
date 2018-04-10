#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.abspath(os.path.join('../../', '06_decision_tree/codes')))
import numpy as np
from decision_tree_classfication import *

class RandomTree(object):
    def __init__(self, num_trees= 5, min_sample_splits=2, min_impurity=1e-7, max_depth=float("inf"),
                 bagging_size = 1e+3, feature_num=5):
        # decision tree parameters
        self.min_sample_splits = min_sample_splits
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        # bagging parameters
        self.num_trees = num_trees
        self.bagging_size = bagging_size # 每棵树训练的样本数量
        self.feature_num = feature_num # 每棵树训练使用的特征个数
        self.trees = []

    def fit(self, X, y):
        samples, features = np.shape(X)
        for i in xrange(self.num_trees):
            sample_idxs = np.random.choice(samples, size=self.bagging_size, replace=True)
            feature_idx = np.random.choice(features, size=self.feature_num, replace=True)
            sample_X = X[sample_idxs, feature_idx]
            sample_y = y[sample_idxs]
            tree = DecisionTreeClassfication(min_sample_splits=self.min_sample_splits, min_impurity=self.min_impurity,
                                             max_depth=self.max_depth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

    def predict(self, X):
        predict_res = []
        for tree in self.trees:
            predict_res.append(tree.predict(X))
        predict_res = np.array(predict_res).T
        res = []
        for i in xrange(predict_res):
            big_cnt = np.bincount(i)
            res.append(np.argmax(big_cnt))
        return res