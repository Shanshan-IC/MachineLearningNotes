#!/usr/bin/env python
# -*- coding: utf-8 -*-


from decision_tree import DecisionTree
from info_cal import *

class DecisionTreeClassfication(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = cal_entropy(y)
        info_gain = entropy - p * cal_entropy(y1) - (1-p) * cal_entropy(y2)
        return info_gain

    def _majority_vote(self, y):
        vote = None
        max_cnt = 0
        for label in np.unique(y):
            cnt = len(y[y == label])
            if cnt > max_cnt:
                max_cnt = cnt
                vote = label
        return vote

    def fit(self, X, y):
        self._impurity_cal = self._calculate_information_gain
        self._leaf_value_cal = self._majority_vote
        super(DecisionTreeClassfication, self).fit(X, y)
