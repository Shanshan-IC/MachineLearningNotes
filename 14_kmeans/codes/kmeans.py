#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/8 11:35
# @Author  : Shanshan Fu
# @File    : KMeans.py  :
# @Contact : 33sharewithu@gmail.com
import numpy as np

"""
算法
选择K个点作为初始质心（随机产生或者从D中选取）
repeat
    将每个点分配到最近的质心，形成K个簇
    重新计算每个簇的质心
until 簇不发生变化或达到最大迭代次数
"""

class KMeans:
    """
    :param n_clusters: 聚类的个数 k
    :param initMethod: 质心初始化的方式，默认是random，或者是指定array
    :param max_iter: 最大的迭代次数
    """
    def __init__(self, n_clusters = 5, initMethod = 'random', max_iter = 100):
        if (hasattr(initMethod, '__array__')):
            n_clusters = initMethod.shape[0]
            self.centroids = np.asarray(initMethod, dtype = np.float)
        else:
            self.centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.initMethod = initMethod

    def __randomCenter(self, X, k):
        features = X.shape[1] # 特征数量
        centroids = np.array([])
        for i in range(features):
            mins = X[:, i].min
            maxs = X[:, i].max
            centroids_i = np.random.uniform(low = mins, high = maxs, size=(k, ))
            centroids = np.append(centroids, centroids_i)
        return centroids

    def __calDistances(self, X):
        return np.linalg.norm(self.centroids[:, np.newaxis] - X, axis=2)

    def fit(self, X):
        # 检查类型
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                TypeError("Not support non-array type!")
        # 初始化簇心
        if (self.initMethod == 'random'):
            self.centroids = self.__randomCenter(X, self.n_clusters)
        samples = X.shape[0]
        self.clusterAssment = np.empty((samples, 2))  # m*2的矩阵，第一列存储样本点所属的族的索引值，
                                                # 第二列存储该点与所属族的质心的平方误差
        for _ in range(self.max_iter):
            for i in range(samples):
                distances = self.__calDistances(X)
                min_idx = distances.argmin(axis = 1)[0]
                self.centroids[min_idx] = (self.centroids[min_idx] * distances)