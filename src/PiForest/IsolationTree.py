# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:49:39 2019

@author: seemandhar
"""
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix

class IsolationTree:
    def __init__(self, height_limit, current_height):

        self.height_limit = height_limit
        self.current_height = current_height
        self.split_by = None
        self.split_value = None
        self.right = None
        self.left = None
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1


    def fit(self, X:np.ndarray):

        if len(X) <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            self.size = X.shape[0]

            return self
        split_by = random.choice(np.arange(X.shape[1]))
        X_col = X[:, split_by]
        min_x = X_col.min()
        max_x = X_col.max()

        if min_x == max_x:
            self.exnodes = 1
            self.size = len(X)

            return self

        else:

            split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)

            w = np.where(X_col < split_value, True, False)
            del X_col

            self.size = X.shape[0]
            self.split_by = split_by
            self.split_value = split_value

            self.left = IsolationTree(self.height_limit, self.current_height + 1).fit(X[w])
            self.right = IsolationTree(self.height_limit, self.current_height + 1).fit(X[~w])
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self






def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0
