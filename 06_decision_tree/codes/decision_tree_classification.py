#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/4 13:12
# @Author  : Shanshan Fu
# @File    : decision_tree_classification.py  :
# @Contact : 33sharewithu@gmail.com

from decision_tree import DecisionTree
class DecisionTreeClassfication(DecisionTree):
    def _majority_vote(self):

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(DecisionTreeClassfication, self).fit(X, y)