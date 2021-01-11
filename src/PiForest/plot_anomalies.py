import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import time
import iforest1
import math
from sklearn.metrics import accuracy_score as ascore
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error,confusion_matrix, f1_score, average_precision_score

def plot_anomalies(X1,X, y,k, sample_size=256, n_trees = 10, desired_TPR=None, percentile = None, normal_ymax=None, bins=20):
    print(n_trees)
    it = iforest.IsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)
    fit_start = time.time()
    it.fit(X1)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, t = roc_curve(y, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    if desired_TPR is not None:
            threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)
    else:
        percentile=None
        threshold = np.percentile(scores, percentile)
    y_pred = it.predict_from_anomaly_scores(scores, threshold=0.8)
    
    print(ascore(y, y_pred))
    yyy.append([roc_auc])