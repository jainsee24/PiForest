# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:37:55 2021

@author: sj
"""
import numpy as np
import pandas as pd
import rrcf
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt
import seaborn as sns

# Read data
taxi = pd.read_csv('nyc_taxi.csv',
                   index_col=0)
print(taxi)
taxi.index = pd.to_datetime(taxi.index)
data = taxi['value'].astype(float).values
print(data)
# Create events
events = {
'independence_day' : ('2014-07-04 00:00:00',
                      '2014-07-07 00:00:00'),
'labor_day'        : ('2014-09-01 00:00:00',
                      '2014-09-02 00:00:00'),
'labor_day_parade' : ('2014-09-06 00:00:00',
                      '2014-09-07 00:00:00'),
'nyc_marathon'     : ('2014-11-02 00:00:00',
                      '2014-11-03 00:00:00'),
'thanksgiving'     : ('2014-11-27 00:00:00',
                      '2014-11-28 00:00:00'),
'christmas'        : ('2014-12-25 00:00:00',
                      '2014-12-26 00:00:00'),
'new_year'         : ('2015-01-01 00:00:00',
                      '2015-01-02 00:00:00'),
'blizzard'         : ('2015-01-26 00:00:00',
                      '2015-01-28 00:00:00')
}
print(taxi)
taxi['event'] = np.zeros(len(taxi))
print(taxi)
for event, duration in events.items():
    start, end = duration
    taxi.loc[start:end, 'event'] = 1
    
    
taxi.to_csv('finalnyc.csv')  


taxi = pd.read_csv('foresdatatime.csv',
                   index_col=0)
data = taxi['Humidity'].astype(float).values
 
taxi.index = pd.to_datetime(taxi.index)

print(taxi)



num_trees = 200
shingle_size = 1
tree_size = 256

# Use the "shingle" generator to create rolling window
points = rrcf.shingle(data, size=shingle_size)
points = np.vstack([point for point in points])
n = points.shape[0]
sample_size_range = (n // tree_size, tree_size)

forest = []
while len(forest) < num_trees:
    ixs = np.random.choice(n, size=sample_size_range,
                           replace=False)
    trees = [rrcf.RCTree(points[ix], index_labels=ix)
             for ix in ixs]
    forest.extend(trees)
    
avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)

for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf)
                        for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
    
avg_codisp /= index
avg_codisp.index = taxi.iloc[(shingle_size - 1):].index


contamination = taxi['event'].sum()/len(taxi)
IF = IsolationForest(n_estimators=num_trees,
                     contamination=contamination,
                     behaviour='new',
                     random_state=0)
IF.fit(points)
if_scores = IF.score_samples(points)
if_scores = pd.Series(-if_scores,
                      index=(taxi
                             .iloc[(shingle_size - 1):]
                             .index))



# =============================================================================
# avg_codisp = ((avg_codisp - avg_codisp.min())
#               / (avg_codisp.max() - avg_codisp.min()))
# =============================================================================
if_scores = ((if_scores - if_scores.min())
              / (if_scores.max() - if_scores.min()))
              
fig, ax = plt.subplots(2, figsize=(10, 6))
(taxi['Humidity'] / 1000).plot(ax=ax[0], color='0.5',
                            alpha=0.8)
if_scores.plot(ax=ax[1], color='#7EBDE6', alpha=0.8,
               label='IF')
avg_codisp.plot(ax=ax[1], color='#E8685D', alpha=0.8,
                label='RRCF')
ax[1].legend(frameon=True, loc=2, fontsize=12)



ax[0].set_xlabel('')
ax[1].set_xlabel('')

ax[0].set_ylabel('Taxi passengers (thousands)', size=13)
ax[1].set_ylabel('Normalized Anomaly Score', size=13)
ax[0].set_title('Anomaly detection on NYC Taxi data',
                size=14)

ax[0].xaxis.set_ticklabels([])

ax[0].set_xlim(taxi.index[0], taxi.index[-1])
ax[1].set_xlim(taxi.index[0], taxi.index[-1])
plt.tight_layout()
plt.savefig('rrcf vs if.pdf')

y=[]
for i in range(0,len(taxi['event'])):
    k=0
    for j in range(i,i+1):
        if taxi['event'][j]==1:
            k=1
    y.append(k)
print(y)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y,avg_codisp)
print('rrcf',auc)
print('len rrcf',len(avg_codisp))

auc = roc_auc_score(y,if_scores)
print('iforest',auc)
print('len ifor',len(if_scores))


