#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left_branch = None, right_branch=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.value = value

class DecisionTree(object):
    def __init__(self, min_sample_splits=2, min_impurity=1e-7, max_depth=float("inf")):
        self.root = None
        self.min_sample_splits = min_sample_splits
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        # 如果是分类问题就是信息增益，回归问题就基尼指数
        self._impurity_cal = None
        self._leaf_value_cal = None #计算叶子
        self.loss = None

    def _divide_features(self, Xy, feature, threshold):
        split_func = None
        if isinstance(threshold, int) or isinstance(threshold, float):
            split_func = lambda s: s[feature] >= threshold
        else:
            split_func = lambda s: s[feature] == threshold
        Xy1 = np.array([s  for s in Xy if split_func(s)])
        Xy2 = np.array([s for s in Xy if not split_func(s)])
        return Xy1, Xy2

    def _build_tree(self, X, y, depth=0):
        if len((np.shape(y))) == 1:
            y = np.expand_dims(y, axis=1)#拉成一列
        n_samples, n_feature = np.shape(X)
        Xy = np.concatenate((X, y), axis=1)
        largest_impurity = 0
        best_criteria = None
        best_sets = None
        if n_samples >= self.min_sample_splits and depth <= self.max_depth:
            #遍历每个特征的每个特征值
            for feature in range(n_feature):
                unique_vals = np.unique(np.expand_dims(X[:, feature], axis=1))
                for threshold in unique_vals:
                    Xy1, Xy2 = self._divide_features(Xy, feature, threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        X1, y1 = Xy1[:, :n_feature], Xy1[:, n_feature:]
                        X2, y2 = Xy2[:, :n_feature], Xy2[:, n_feature:]
                        impurity = self._impurity_cal(y, y1, y2)
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {'feature': feature, 'threshold': threshold}
                            best_sets = {'leftX': X1,
                                         'leftY': y1,
                                         'rightX': X2,
                                         'rightY': y2}
        if largest_impurity > self.min_impurity:
            left_node = self._build_tree(best_sets['leftX'], best_sets['leftY'], depth+1)
            right_node = self._build_tree(best_sets['rightX'], best_sets['rightY'], depth+1)
            return DecisionNode(feature = best_criteria['feature'], threshold = best_criteria['threshold'],
                                left_branch= left_node, right_branch=right_node)
        leaf_value = self._leaf_value_cal(y)
        return DecisionNode(value=leaf_value)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict_val(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature]
        branch = tree.right_branch
        if isinstance(feature_val, int) or isinstance(feature_val, float):
            if feature_val >= tree.threshold:
                branch = tree.left_branch
        elif feature_val == tree.threshold:
            branch = tree.left_branch
        return self.predict_val(x, branch)

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_val(x, self.root))
        return y_pred