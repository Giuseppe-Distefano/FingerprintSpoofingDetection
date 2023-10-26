################################
#     MODULES IMPORTATIONS     #
################################
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import modules.utility as utility
import modules.plots as plots
import modules.dataset as dataset


####################
#     FUNCTIONS    #
####################
# ----- Principal Component Analysis -----
def apply_pca (D, L, m, output_folder=None):
    # Identify the largest eigenvalues and eigenvectors over to which we project original data
    _,U = npla.eigh(utility.compute_covariance(D))
    P = U[:, ::-1][:, 0:m]
    
    if output_folder!=None:
        # Project data
        DP = np.dot(P.T, D)
        plots.plot_pca_scatters(DP, L, output_folder)

    return P


# ----- Linear Discriminant Analysis -----
def apply_lda (D, L, m, output_folder=None):
    # Compute between- and within-class covariance matrices
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    diff0 = utility.compute_mean(D0) - utility.compute_mean(D)
    diff1 = utility.compute_mean(D1) - utility.compute_mean(D)
    SB = np.outer(diff0,diff0) + np.outer(diff1,diff1)
    SW = utility.compute_covariance(D0) + utility.compute_covariance(D1)
    
    # Identify the m largest eigenvalues and eigenvectors over to which we project original data
    U,s,_ = npla.svd(SW)
    P1 = np.dot(U, utility.row_to_column(1.0/s**0.5)*U.T)
    SBt = np.dot(P1, np.dot(SB, P1.T))
    U,_,_ = npla.svd(SBt)
    P2 = U[:, 0:m]
    W = np.dot(P1.T, P2)

    if output_folder!=None:
        # Project data
        DP = np.dot(W.T, D)
        plots.plot_lda_histograms(DP, L, output_folder)
