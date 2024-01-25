################################
#     MODULES IMPORTATIONS     #
################################
import modules.utility as utility
import numpy


#####################
#     FUNCTIONS     #
#####################
# ----- Center data -----
def center_data (D):
    m = D.mean(1)
    return D-utility.row_to_column(m)


# ----- Standardize data -----
def standardize_data (D):
    sd = numpy.sqrt(numpy.var(D, axis=1))
    s = numpy.zeros(D.shape)
    for i in range(0, D.shape[1]):
        s[:,i] = D[:,i] / sd
    return s


# ----- ZNorm -----
def znorm (D):
    c = center_data(D)
    return standardize_data(c)


# ----- Whiten data -----
def whiten_data (D):
    _,cov = utility.compute_covariance_matrix(D)
    U,s,V = numpy.linalg.svd(cov)
    inv = numpy.diag(1.0/numpy.sqrt(s))
    wm = numpy.dot(numpy.dot(inv, U.T), V)
    wc = numpy.dot(wm, numpy.dot(cov, wm.T))

    w = numpy.zeros(D.shape)
    for i in range(0, D.shape[1]):
        w[:,i] = numpy.dot(D[:,i], wc)
    return w


# ----- L2Norm -----
def l2norm (D):
    d = numpy.zeros(D.shape)
    for i in range(0, D.shape[1]):
        d[:,i] = D[:,i] / numpy.linalg.norm(D[:,i])
    return d
