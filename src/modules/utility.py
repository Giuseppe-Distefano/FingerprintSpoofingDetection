################################
#     MODULES IMPORTATIONS     #
################################
import numpy
import numpy.linalg
import scipy.special


####################
#     FUNCTIONS    #
####################
# ----- Convert an array into a column -----
def row_to_column (array):
    return array.reshape((array.size, 1))


# ----- Convert an array into a row -----
def column_to_row (array):
    return array.reshape((1, array.size))


# ----- Compute mean -----
def compute_mean (data):
    return row_to_column(data.mean(1))


# ----- Compute covariance -----
def compute_covariance (data):
    mu = compute_mean(data)
    dc = data - mu.reshape((mu.size, 1))
    cov = numpy.dot(dc, dc.T) / float(data.shape[1])
    return cov


# ----- Compute Pearson correlation -----
def compute_correlation(x, y):
    x1_sum, y1_sum = numpy.sum(x), numpy.sum(y)
    x2_sum, y2_sum = numpy.sum(x**2), numpy.sum(y**2)
    cp_sum = numpy.sum(x * y.T)
    n = x.shape[0]

    num = n*cp_sum - x1_sum*y1_sum
    den = numpy.sqrt((n*x2_sum - x1_sum**2) * (n*y2_sum - y1_sum**2))
    return (num/den)


# ----- Compute logarithm of probability density function of a Gaussian distribution -----
def logpdf_GAU_ND (X, mu, C):
    _,det = numpy.linalg.slogdet(C)
    inv = numpy.linalg.inv(C)

    term1 = -0.5 * X.shape[0] * numpy.log(2*numpy.pi)
    term2 = -0.5 * det
    term3 = -0.5 * numpy.dot((X-mu).T, numpy.dot(inv, (X-mu))).sum(0)

    return (term1+term2+term3)


# ----- Classify samples matching log-likelihood ratio and a threshold -----
def predict_labels (llr, threshold):
    predicted = numpy.zeros(len(llr))
    for i in range(len(llr)):
        if (llr[i]>threshold): predicted[i] = 1
    return predicted


# ----- Count the number of mispredictions -----
def count_mispredictions (predicted, LTE):
    wrong = 0
    for i in range(len(LTE)):
        if (predicted[i]!=LTE[i]): wrong += 1
    return wrong


# ----- Square a matrix and transpose it -----
def square_and_transpose (matrix):
    x = matrix[:,None]
    xxT = x.dot(x.T).reshape((x.size)**2, order='F')
    return xxT


# ----- Compute ZNorm -----
def compute_znorm (DTR, DTE):
    mu_r = row_to_column(DTR.mean(1))
    sigma_r = row_to_column(DTR.std(1))
    normalized_r = (DTR-mu_r) / sigma_r
    normalized_e = (DTE-mu_r) / sigma_r
    return normalized_r, normalized_e


# ----- Randomize data -----
def randomize (D, L, seed):
    numpy.random.seed(seed)
    indexes = numpy.random.permutation(D.shape[1])
    d = numpy.zeros((10, D.shape[1]))
    l = numpy.zeros((L.size,))
    i = 0
    for ind in indexes:
        d[:,i] = D[:,ind]
        l[i] = L[ind]
        i += 1
    return d, l