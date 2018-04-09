import math
import numpy as np

def cal_entropy(y):
    unique_y = np.unique(y)
    entropy = 0
    for label in unique_y:
        cnt = len(y[y == label])
        p = cnt / len(y)
        entropy += -p * np.log2(p)
    return entropy