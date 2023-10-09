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
# ----- Apply Principal Component Analysis -----
def apply_pca (D, L, m):
    output_folder = "../output/02_PCA"

    s,U = npla.eigh(utility.compute_covariance(D))
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    plots.plot_scatters(DP, L, m, output_folder)


# ----- Apply Linear Discriminant Analysis -----
def apply_lda (D, L, m):
    output_folder = "../output/03_LDA"

    # Computation of between- and within-class covariance matrices
    N = D.shape[1]
    mu = utility.compute_mean(D)
    SB = 0
    SW = 0
    for i in range(dataset.distinct_classes):
        Di = D[:, L==i]
        ni = Di.shape[1]
        mui = utility.compute_mean(Di)
        SB += ni/N * np.outer(mui-mu, mui-mu)
        SW += ni/N * utility.compute_covariance(Di)
    
    # Projection of data over the new set of directions
    s,U = spla.eigh(SB,SW)
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = spla.svd(W)
    DP = np.dot(W.T, D)

    # Plot projections
    plots.plot_lda_histograms(DP, L, 40, output_folder)
    plots.plot_scatters(DP, L, m, output_folder)