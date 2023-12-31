import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def vcol(v):   
    return v.reshape((v.size, 1))
def vrow(v):
    return v.reshape((1, v.size)) 
#LOAD 
def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            v=line.split(",")[0:10]
            attr= vcol(np.array(v, dtype=np.float64))
            DList.append(attr)
            labelsList.append(line.split(",")[10])
            
    D=np.hstack(DList)  
    L=np.asarray(labelsList, dtype=np.int32)
    return D,L

def computeMean(D):
    return vcol(D.mean(1))

def computeCovarianceMatrix(D):
    mu = computeMean(D)
    C = np.dot((D-mu), (D-mu).T) / D.shape[1]
    return C

##### Compute Pearson correlation #####
def compute_correlation(x, y):
    x1_sum = np.sum(x)
    y1_sum = np.sum(y)
    x2_sum = np.sum(x**2)
    y2_sum = np.sum(y**2)
    cross_product_sum = np.sum(x * y.T)
    n = x.shape[0]

    num = n*cross_product_sum - x1_sum*y1_sum
    den = np.sqrt((n*x2_sum - x1_sum**2) * (n*y2_sum - y1_sum**2))
    return num/den
