#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class MultinomialNB(object):

    def __init__(self, alpha):
        self.alpha = alpha
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)