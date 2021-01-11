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
def pca(df_data,k):
    df_data=df_data.dropna()
    X_train=df_data
    X_train=StandardScaler().fit_transform(X_train)
    
    #print(X_train)
    pca=PCA(n_components=k)
    #print(pca)
    X_train=pca.fit_transform(X_train)
    
    X_train=pd.DataFrame(X_train)
    aa=[]
    for i in range(1,k+1):
        aa.append("fe"+str(i))
    X_train.columns=aa
    #print(X_train,X_test)
    for i in df_data:
        df_data=df_data.drop(i,1)
    
    df_data = pd.concat([df_data, X_train], axis=1, sort=False)
    df_data=df_data.dropna()
    return df_data
    