#!/usr/bin/env python
# -*- coding: utf-8 -*-

# kd tree的node结构
class node:
    def __init__(self, val, split, left = None, right = None):
        self.left = left
        self.right = right
        self.val = val
        self.split = split

# kd tree
class tree:
    def __index__(self, depth):
        self.depth = depth

    def get_median(self, list):
        m = len(list) / 2
        return m, list[m]

    def build_tree(self, data, d):
        if (len(data) == 0):
            return
        data = sorted(data, key=lambda x:x[d])
        m, point = self.get_median(data)
        tree = node(val = point, split = m)
        tree.left = self.build_tree(data[,0: m], d+1 % self.depth)
        tree.right = self.build_tree(data[, m+1:], d+1 % self.depth)
        return tree

    def search_tree(self, tree, target, d):
        if target[d] < tree.val[d]:
            if (tree.left != None):
                return self.search_tree(tree.left, target, d+1)
        else:
            if (tree.right != None):
                return self.search_tree(tree.right, target, d+1)

