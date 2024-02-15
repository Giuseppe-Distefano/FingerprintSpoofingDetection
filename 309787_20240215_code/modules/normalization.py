##### LIBRARIES #####
import utils as utils
import numpy


##### FUNCTIONS #####
# --- Remove mean from data ---
def center_data (X):
    return X - utils.vcol(X.mean(1))


# --- Apply Z-Norm over data ---
def znorm_data (X):
    return standardize_data(center_data (X))


# --- Data whitening ---
def whiten_data (X):
    _,cov = utils.compute_covariance(X)
    U, s, V = numpy.linalg.svd(cov)
    inv_sqrt_s = numpy.diag(1.0 / numpy.sqrt(s))
    whitening_matrix = numpy.dot(inv_sqrt_s, U.T)
    whitening_matrix = numpy.dot(whitening_matrix, V)
    whitened_cov = numpy.dot(whitening_matrix, numpy.dot(cov, whitening_matrix.T))
    
    W_DTR = numpy.zeros(X.shape) 
    for i in range(0,X.shape[1]): W_DTR[:,i] = numpy.dot(X[:,i], whitened_cov)
    return W_DTR


# --- Standardize data ---
def standardize_data (X):
    standard_deviation = numpy.sqrt(numpy.var(X, axis=1))
    S_DTR = numpy.zeros(X.shape) 
    for i in range(0,X.shape[1]): S_DTR[:,i] = X[:,i] / standard_deviation
    return S_DTR


# --- ___ ---
def l2NormalizingData(X):
    L2_DTR = numpy.zeros(X.shape) 
    for i in range(0,X.shape[1]): L2_DTR[:,i] = X[:,i]/numpy.linalg.norm(X[:,i])
    return L2_DTR
