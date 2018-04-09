#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/9 13:24
# @Author  : Shanshan Fu
# @File    : decision_tree_regression.py  :
# @Contact : 33sharewithu@gmail.com

from decision_tree import DecisionTree
from info_cal import *

class DecisionTreeRegression(DecisionTree):
    def cal_variance_reduction(self, y, y1, y2):
        var_total = cal_variance(y)
        var_1 = cal_variance(y1)
        var_2 = cal_variance(y2)
        variance_reduce = var_total - (len(y1) / len(y) * var_1 + len(y2)/len(y) * var_2)
        return sum(variance_reduce)

    def mean_y(self, y):
        val = np.mean(y, axis=0)
        return val if len(val) > 1 else val[0]

    def fit(self, X, y):
        self._impurity_cal = self.cal_variance_reduction
        self._leaf_value_cal = self.mean_y
        super(DecisionTreeRegression).fit(X, y)
