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

def cal_variance(y):
    mean = np.ones(np.shape(y)) * y.mean(0) # 生成和y结构相同的均值矩阵
    n_samples = np.shape(y)[0]
    variance = (1 / n_samples) * np.diag((y-mean).T.dot(y - mean))
    return variance