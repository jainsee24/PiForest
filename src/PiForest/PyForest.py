# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:13:40 2021

@author: sj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:30:20 2020

@author: sj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:39:19 2019

@author: sj
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import time
import IsolationTree
import math
from sklearn.metrics import accuracy_score as ascore
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error,confusion_matrix, f1_score, average_precision_score

yyy=[]
yyy1=[]


def find_TPR_threshold(y, scores, desired_TPR):
    threshold = 1
    while threshold > 0:
        y_pred = [1 if p >= threshold else 0 for p in scores]
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred,labels=[0,1]).ravel()
        if (tp + fn)==0:
            return 1,0
        TPR = tp / (tp + fn)
        FPR = fp / (fp + tn)
        if TPR >= desired_TPR:
            return threshold, FPR

        threshold = threshold - 0.001
    
    return threshold, FPR


if __name__ == '__main__': 
        k=4
        targetcol = "Class"
        
        
        zz=["arduinoRealWorldDatasetGenerated.csv"]
        desired_TPR = 99
        desired_TPR /= 100.0
        
        X = pd.read_csv(zz[0],encoding = 'unicode_escape')
        X, y11 = X.drop(targetcol, axis=1), X[targetcol]
        X=pca(X,2)
        aucc=[]
        n_trees = 10
        label = []
        sample_size = 256
        neg = []
        win = []
        count = 0
        c=[]
        score=[]
        win = pd.DataFrame(win)
        for i in range(sample_size):
            win = win.append(X.iloc[[i]])
        it=IsolationTree.IsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)
        fit_start = time.time()
        it.fit(win)
        fit_stop = time.time()
        fit_time = fit_stop - fit_start
        #print("fit time is for stating ",10," trees: ",fit_time)
        
        win=[]
        
        
        win = pd.DataFrame(win)
        ll=[]
        ypred=[]
        for i in range(0,sample_size):
            ypred.append(0)
        scoretime=0
        y2=[]
        y21=[]
        anomalys=[]
        anomalys1=[]
        fpr1=[]
        tpr1=[]
        import time
        a=time.time()
        for i in range(sample_size,len(X)):
            win = win.append(X.iloc[i])
            win1=[]
            win1= pd.DataFrame(win1)
            win1 = win1.append(X.iloc[i])
            fit_start = time.time()
            ppp=it.anomaly_score(win1,sample_size).tolist()[0][0]
            fit_stop = time.time()
            scoretime+=fit_stop-fit_start
            anomalys.append(ppp)
            anomalys1.append(ppp)
            y2.append(y11.iloc[i])
            y21.append(y11.iloc[i])
            ll.append(y11.iloc[i])
            if len(win)==sample_size:
                it.deletektrees(k)
                it.insertktrees(win,k)
                win=[]
                win=pd.DataFrame(win)
                if desired_TPR is not None:
                   
                    threshold, FPR = find_TPR_threshold(y2, anomalys, desired_TPR)
                else:
                    percentile=Noness
                    threshold = np.percentile(anomalys, percentile)
                y_pred = it.predict_from_anomaly_scores(anomalys, threshold=0.5)
                