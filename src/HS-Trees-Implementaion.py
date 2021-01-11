import pandas as pd
from skmultiflow.anomaly_detection import HalfSpaceTrees
from sklearn import metrics
import numpy as np


fp=0
tp=0
coords = [(0,0)]

import csv 
  
# csv file name 
filename = "breastw.csv"
  
fields = [] 
rows = [] 
  
with open(filename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    fields = next(csvreader) 
    for row in csvreader: 
        rows.append(row) 

X=[]
yl=[]
for i in rows:
    x=[]
    for j in range(0,len(i)-1):
        x.append(int(i[j]))
    X.append(x.copy())
    yl.append(int(i[-1]))
pos_count =0
neg_count=0
for i in yl:
    if i==0:
        neg_count+=1
    else:
        pos_count+=1
# Train the estimator(s) with the samples provided by the data stream
import time as t
a=t.time()
half_space_trees = HalfSpaceTrees(random_state=1)
y2=[]
score=[]
for i in range(0,len(X)):
    print(i)
    X1, y = X[i],yl[i]
    X1=np.array([X1])
    yy=np.array([y])
    score.append(half_space_trees.predict(X1)[0])
    y_pred=half_space_trees.predict(X1)
    print(half_space_trees.predict_proba(X1))
    y2.append(y_pred[0])
    if y == 1:
        if y_pred[0] == 1:
            tp+=1
        else:
            fp+=1
    coords.append((fp, tp))
    half_space_trees.partial_fit(X1, yy)



fp, tp = map(list, zip(*coords))
for i in range(0,len(fp)):
    fp[i]/=neg_count
for i in range(0,len(tp)):
    tp[i]/=pos_count

b=t.time()
print(b-a)
print(fp)
print(tp)

import matplotlib.pyplot as plt


plt.scatter(fp, tp)
plt.savefig('1.pdf')
print(metrics.auc(fp, tp))


fpr, tpr, thresholds = metrics.roc_curve(yl, score, pos_label=1)
print(metrics.auc(fpr, tpr))



import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % metrics.auc(fpr, tpr))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('1.pdf')