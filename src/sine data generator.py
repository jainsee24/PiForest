# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:26:43 2021

@author: sj
"""

import numpy as np
import matplotlib.pyplot as plt

# Generating time data using arange function from numpy
time = np.arange(0, 800)

# Finding amplitude at each time
amplitude =[]
a=20
x=len(time)/a
for i in time:
    amplitude.append(np.sin(i/a))

xx=amplitude[490]
a=[0]*len(amplitude)
for i in range(500,520):
    amplitude[i]=xx
    a[i]=1
for i in range(0,len(amplitude)):
    amplitude[i]+=1
    amplitude[i]*=40
# Plotting time vs amplitude using plot function from pyplot
plt.plot(time, amplitude)
print(amplitude)
# Settng title for the plot in blue color
plt.title('Sine Wave', color='b')

# Setting x axis label for the plot
plt.xlabel('Time'+ r'$\rightarrow$')

# Setting y axis label for the plot
plt.ylabel('Sin(time) '+ r'$\rightarrow$')

# Highlighting axis at x=0 and y=0
# =============================================================================
# plt.axhline(y=0, color='k')
# plt.axvline(x=0, color='k')
# =============================================================================

# Finally displaying the plot
plt.savefig('plotsin.pdf')
l=[]
for i in range(0,len(amplitude)):
    l.append([amplitude[i],a[i]])
import numpy as np
import pandas as pd
data = np.array(l)

dataset = pd.DataFrame({'f1': data[:, 0], 'label': data[:, 1]})
dataset.to_csv('sine.csv',index=False)  