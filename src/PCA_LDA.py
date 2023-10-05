import numpy as np
import utility as u
import scipy as sc

#PCA
def computeEigenValuesAndEigenVectors(C, D, m): 
    #WE COMPUTE EIGEN-DECOMPOSITION
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    
    DP = np.dot(P.T, D)
    #DP = Data Projection
    return DP

#LDA
def compute_sw(D, L): #WITHIN CLASS COVARIANCE
    SW = 0
    for i in [0, 1]: #SINCE WE KNOW THAT WE HAVE 2 CLASSES
        SW += (L==i).sum() * u.computeCovarianceMatrix(D[:, L==i]) 
    SW = SW / D.shape[1] #D.shape[1] IS THE NUMBER OF SAMPLES OF THE WHOLE DATASET
    print(SW)
    return SW


def compute_sb(D, L):
    SB = 0
    muG = u.computeMean(D) #DATASET MEAN (THE G STANDS FOR "GLOBAL")
    for i in set(list(L)):  
        X = D[:, L==i]
        mu = u.computeMean(X) #MEAN OF A SPECIFIC CLASS
        SB += X.shape[1] * np.dot((mu - muG), (mu - muG).T) #X.shape[1] GIVES US THE NUMBER OF ELEMENTS OF THE CURRENT CLASS --> SCRIVERE (L==i).sum() E D[:, L==i].shape[1]
    return SB / D.shape[1]
        

def compute_LDA_directions1(D,L, m):
    SW = compute_sw(D, L) #WITHIN CLASS COVARIANCE MATRIX
    SB = compute_sb(D, L) # between class covariance matrix

    s, U = sc.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    
    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]
    
    DP = np.dot(W.T, D) #AS FOR THE PCA WE apply the projection to a matrix of samples D:
    return DP
