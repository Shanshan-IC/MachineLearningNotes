#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 11:58
# @Author  : Shanshan Fu
# @File    : linear_regression_sgd.py  :
# @Contact : 33sharewithu@gmail.com

import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('../..', '31_scaler/codes')))
from min_max_scaler import *

class LinearRegressionSGD:
    def __init__(self, alpha = 0.1, max_iter = 100, scaler=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.scaler = scaler

    def fit(self, X, y):
        self.X = MinMaxScaler().fit(X) if self.scaler else X



