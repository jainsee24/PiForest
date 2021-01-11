# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:44:26 2021

@author: sj
"""

# -*- coding: utf-8 -*-import numpy as np
import rrcf
import pandas as pd

# Generate data
zz=["sat.csv"]
targetcol = "label"    
X = pd.read_csv(zz[0],encoding = 'unicode_escape')

X,Y = X.drop(targetcol, axis=1), X[targetcol]   
sin=[]
c='attr1	attr2	attr3	attr4	attr5	attr6	attr7	attr8	attr9	attr10	attr11	attr12	attr13	attr14	attr15	attr16	attr17	attr18	attr19	attr20	attr21	attr22	attr23	attr24	attr25	attr26	attr27	attr28	attr29	attr30	attr31	attr32	attr33	attr34	attr35	attr36'
c=c.split('	')
print(c)
for i in range(0,len(X)):
    x=[]
    for j in range(0,len(c)):
        x.append(X[c[j]][i])
    sin.append(x)
print(sin)
# Set tree parameters
num_trees = 10
shingle_size = 1
tree_size = 256
# Create a forest of empty trees
forest = []
import time
a=time.time()
for _ in range(num_trees):
    forest.append(rrcf.RCTree())
 
# Use the "shingle" generator to create rolling window
points = rrcf.shingle(sin, size=shingle_size)

# Create a dict to store anomaly score of each point
avg_codisp = {}

# For each shingle...
for index, point in enumerate(points):
    print(index)
    # For each tree in the forest...
    for tree in forest:
        # If tree is above permitted size, drop the oldest point (FIFO)
        if len(tree.leaves) > tree_size:
            tree.forget_point(index - tree_size)
        # Insert the new point into the tree
        tree.insert_point(point, index=index)
        # Compute codisp on the new point and take the average among all trees
        if not index in avg_codisp:
            avg_codisp[index] = 0
        avg_codisp[index] += tree.codisp(index) / num_trees
        
      
l=[]
b=time.time()
print('Time: ',b-a)
for i in range(0,len(avg_codisp)):
    l.append(avg_codisp[i])      
      
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(Y[:len(l)], l)
print(auc)

"""
Created on Fri Apr 10 18:34:11 2020

@author: sj
"""

