#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/4 13:12
# @Author  : Shanshan Fu
# @File    : decision_tree_classification.py  :
# @Contact : 33sharewithu@gmail.com

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
                vote = y
        return vote

    def fit(self, X, y):
        self._impurity_cal = self._calculate_information_gain
        self._leaf_value_cal = self._majority_vote
        super(DecisionTreeClassfication, self).fit(X, y)


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = datasets.load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
clf = DecisionTreeClassfication()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)