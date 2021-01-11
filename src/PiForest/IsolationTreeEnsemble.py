import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):

        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = np.log2(sample_size)
        self.trees = []
    def deletektrees(self,k):
        for i in range(k):
            self.trees.pop(0)
        
    def insertktrees(self,win,k):
        if isinstance(win, pd.DataFrame):
            win = win.values
            len_x = len(win)
        
        for i in range(k):
            sample_idx = random.sample(list(range(len_x)), self.sample_size)
            temp_tree = IsolationTree(self.height_limit, 0).fit(win[sample_idx, :])
            self.trees.append(temp_tree)
       
    def fit(self, X:np.ndarray):

        if isinstance(X, pd.DataFrame):
            X = X.values
            len_x = len(X)
            self.trees = []

    
        for i in range(self.n_trees):
            sample_idx = random.sample(list(range(len_x)), self.sample_size)
            
            temp_tree = IsolationTree(self.height_limit, 0).fit(X[sample_idx, :])
            self.trees.append(temp_tree)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        pl_vector = []
        if isinstance(X, pd.DataFrame):
            X = X.values

        for x in (X):
            pl = np.array([path_length_tree(x, t, 0) for t in self.trees])
            pl = pl.mean()

            pl_vector.append(pl)
        pl_vector = np.array(pl_vector).reshape(-1, 1)
        return pl_vector

    def anomaly_score(self, X:np.ndarray,yy) -> np.ndarray:
        return 2.0 ** (-1.0 * self.path_length(X) / c(yy))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:

        predictions = [1 if p >= threshold else 0 for p in scores]

        return predictions

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."

        scores = 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))
        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions
