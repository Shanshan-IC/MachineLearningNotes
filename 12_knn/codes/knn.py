#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/10 14:29
# @Author  : Shanshan Fu
# @File    : KNN.py  :
# @Contact : 33sharewithu@gmail.com

# s算法
# 计算已知类别数据集中的点与当前点的距离
# 按照距离递增次序排序
# 选取与当前点距离最小的k个点
# 确定前k个点所在类别的出现概率，返回前k个点出现频率最高的类别作为当前点的预测分类

import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import normalize

import os, sys
sys.path.append(os.path.abspath(os.path.join('../..', 'utils')))

class KNN:
    def __init__(self, n_neighbors = 2, norm = False, metrics = 'euclidean', mode = 'auto'):
        self.n_neighbors = n_neighbors
        self.metrics = metrics
        self.norm = norm
        self.mode = mode


    def fit(self, train_X, test_X):
        train_X = self.autoNorm(train_X)
        test_X = self.autoNorm(test_X)
        distance_matrix = self.get_distance_matrix(test_X, train_X)
        self.distance_rank = np.argsort(distance_matrix)
        return self

    def predict(self, train_X, train_Y, test):
        if (self.n_neighbors > train_X.shape[1]):
            self.n_neighbors = train_X.shape[1]
        test_Y = []
        for i in range(0, test.shape[0]):
            list = self.distance_rank[i, 0:self.n_neighbors]
            list_Y = train_Y[list]
            unique, counts = np.unique(list_Y, return_counts=True)
            max_index = counts.argmax()
            test_Y.append(unique[max_index])
        return np.array(test_Y)

    # normalize dataset
    def autoNorm(self, X):
        return normalize(X, axis=0, norm=self.norm)

    def get_distance_matrix(self, train, test):
        return distance.cdist(train, test, self.metrics)

