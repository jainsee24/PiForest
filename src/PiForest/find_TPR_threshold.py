import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
def find_TPR_threshold(y, scores, desired_TPR):
    threshold = 1

    while threshold > 0:
        y_pred = [1 if p[0] >= threshold else 0 for p in scores]
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        TPR = tp / (tp + fn)
        FPR = fp / (fp + tn)
        if TPR >= desired_TPR:
            return threshold, FPR

        threshold = threshold - 0.001

    return threshold, FPR