#!/usr/bin/env python
# -*- coding: utf-8 -*-

class MinMaxScaler():
    def __init__(self, feature_range = (0, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range

    def fit(self, X):
        self.min = X.min(axis = 0)
        self.max = X.max(axis = 0)

    def transform(self, X):
        X_std = (X - self.min) / (self.max - self.min)
        (mins, maxs) = self.feature_range
        X_scaled = X_std * (maxs - mins) + mins
        return X_scaled