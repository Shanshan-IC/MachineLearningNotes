#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/12 16:17
# @Author  : Shanshan Fu
# @File    : PCA.py  :
# @Contact : 33sharewithu@gmail.com

import numpy as np

class PCA:
    def __init__(self, n_components = 2):
        self.n_components = n_components

    def fit(self, X, Y = None):
        X_new, X_mean = self.zero_mean(X)
        X_cov = np.cov(X_new, rowvar = 0 )  # 获得协方差矩阵
        X_eig_val, X_eig_vec = np.linalg.eig(X_cov) # 获得特征根和特征向量
        X_eig_index = np.argsort(-X_eig_val)# 对特征根进行从大到小的排序
        X_eig_index_top = X_eig_index[0: self.n_components] #选择top的下标
        X_eig_vec_top = X_eig_vec[:, X_eig_index_top]
        X_transform = np.dot(X_new, X_eig_vec_top)
        X_recon = np.dot(X_transform, np.transpose(X_eig_vec_top)) + X_mean
        return X_transform, X_recon

    # 均值做normalization
    def zero_mean(self, X):
        X_mean = np.mean(X, axis = 0)
        X_new = X - X_mean
        return X_new, X_mean