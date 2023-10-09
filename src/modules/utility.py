##### MODULES IMPORTATIONS #####
import numpy as np
import scipy.linalg as spla
import modules.plots as plots
import numpy.linalg as npla
import scipy.special as spec
import modules.dataset as dataset


##### Convert an array into a column #####
def row_to_column (array):
    return array.reshape((array.size, 1))


##### Convert an array into a row #####
def column_to_row (array):
    return array.reshape((1, array.size))


##### Compute mean #####
def compute_mean (data):
    return data.mean(1)


##### Compute covariance #####
def compute_covariance (data):
    mu = compute_mean(data)
    dc = data - mu.reshape((mu.size, 1))
    cov = np.dot(dc, dc.T) / float(data.shape[1])
    return cov


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


##### Compute means and covariances for each class #####
def compute_muc_sigmac (D, L):
    means = []
    covs = []
    for i in range(dataset.distinct_classes):
        Di = D[:,L==i]
        mu_i = row_to_column(compute_mean(Di))
        cov_i = compute_covariance(Di)
        means.append(mu_i)
        covs.append(cov_i)
    return np.array(means), np.array(covs)


##### Compute means for each class and tied covariance #####
def compute_muc_sigma (D, L):
    means = []
    covs = []
    for i in range(dataset.distinct_classes):
        Di = D[:,L==i]
        mu_i = row_to_column(compute_mean(Di))
        cov_i = compute_covariance(Di)
        means.append(mu_i)
        covs.append(cov_i)
    return np.array(means), np.array(covs)


# ----- Compute logarithm of probability density function of a Gaussian distribution -----
def logpdf_GAU_ND (X, mu, C):
    _,det = npla.slogdet(C)
    inv = npla.inv(C)
    Y = []

    M = X.shape[0]
    term1 = -0.5 * M * np.log(2*np.pi)
    term2 = -0.5 * det
    term3 = -0.5 * ((X-mu) * np.dot(inv, (X-mu))).sum(0)
    Y.append(term1+term2+term3)
    
    return np.array(Y).ravel()


##### Compute likelihoods #####
def compute_likelihoods (D, L, means, covs):
    logS = np.array([logpdf_GAU_ND(D, means[0], covs[0]), logpdf_GAU_ND(D, means[1], covs[1])])
    return logS


##### Compute likelihoods for each sample for each class, considering one unique covariance matrix #####
def compute_tied_likelihoods (D, L, means, covs):
    logS = np.array([logpdf_GAU_ND(D, means[0], covs), logpdf_GAU_ND(D, means[1], covs)])
    return logS


##### Compute class-posterior distributions #####
def compute_logposterior (logS, Pc):
    logSJoint = logS + np.log(Pc)
    logSMarginal = spec.logsumexp(logSJoint, axis=0)
    logSPost = logSJoint - logSMarginal
    return logSPost


##### Evaluate accuracy and error rate #####
def compute_correct_predictions (SPost, LTE):
    predictions = SPost.argmax(axis=0)==LTE
    correctly = predictions.sum()
    incorrectly = predictions.size - correctly
    return correctly


##### Logistic Regression objective #####
def lr_obj_wrap (DTR, LTR, lam):
    def lr_obj (v):
        w,b = v[0:-1], v[-1]
        N = DTR.shape[1]
        term1 = lam/2 * npla.norm(w)**2
        term2 = 0
        #term3 = 0
        for i in range(N):
            ci = LTR[i]
            zi = 2*ci-1
            xi = DTR[:,i]
            term2 += np.logaddexp(0, -zi*(b + np.dot(w.T, xi)))
        loss = term1 + term2/N
        return loss
    return lr_obj


##### Compute scores of Logistc Regression #####
def lr_compute_scores (DTE, LTE, v):
    w,b = v[0:-1], v[-1]
    n = DTE.shape[1]
    LP = []
    for i in range(0,n):
        xt = DTE[:,i]
        si = b + np.dot(w.T, xt)
        if (si>0): LP.append(1)
        else: LP.append(0)
    correctly = np.sum(np.array(LP==LTE))
    return correctly
