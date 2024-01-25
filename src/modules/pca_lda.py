################################
#     MODULES IMPORTATIONS     #
################################
import numpy
import numpy.linalg
import modules.utility as utility
import matplotlib.pyplot


####################
#     FUNCTIONS    #
####################
# ----- Principal Component Analysis -----
def apply_pca (D, L, m, output_folder=None):
    # Identify the largest eigenvalues and eigenvectors over to which we project original data
    _,U = numpy.linalg.eigh(utility.compute_covariance(D))
    P = U[:, ::-1][:, 0:m]
    
    if output_folder!=None:
        # Project data
        DP = numpy.dot(P.T, D)
        plot_pca_scatters(DP, L, output_folder)

    return P


# ----- Linear Discriminant Analysis -----
def apply_lda (D, L, m, output_folder=None):
    # Compute between- and within-class covariance matrices
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    diff0 = utility.compute_mean(D0) - utility.compute_mean(D)
    diff1 = utility.compute_mean(D1) - utility.compute_mean(D)
    SB = numpy.outer(diff0,diff0) + numpy.outer(diff1,diff1)
    SW = utility.compute_covariance(D0) + utility.compute_covariance(D1)
    
    # Identify the m largest eigenvalues and eigenvectors over to which we project original data
    U,s,_ = numpy.linalg.svd(SW)
    P1 = numpy.dot(U, utility.row_to_column(1.0/s**0.5)*U.T)
    SBt = numpy.dot(P1, numpy.dot(SB, P1.T))
    U,_,_ = numpy.linalg.svd(SBt)
    P2 = U[:, 0:m]
    W = numpy.dot(P1.T, P2)

    if output_folder!=None:
        # Project data
        DP = numpy.dot(W.T, D)
        plot_lda_histograms(DP, L, output_folder)


# ----- Plot scatters of data projected after PCA -----
def plot_pca_scatters (D, L, output_folder):
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    matplotlib.pyplot.figure()
    matplotlib.pyplot.scatter(D0[0,:], D0[1,:], label="Spoofed")
    matplotlib.pyplot.scatter(D1[0,:], D1[1,:], label="Authentic")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig("%s/scatter_pca.png" % (output_folder))
    matplotlib.pyplot.close()


# ----- Plot histograms of data projected after LDA -----
def plot_lda_histograms (D, L, output_folder):
    # Split according to label
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    # Plot histograms for each feature
    matplotlib.pyplot.figure()
    matplotlib.pyplot.hist(D0[0,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Spoofed")
    matplotlib.pyplot.hist(D1[0,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Authentic")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig("%s/histogram_lda.png" % (output_folder))
    matplotlib.pyplot.close()


# ----- Plot scatters of data projected after LDA -----
def plot_lda_scatters (D, L, output_folder):
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    matplotlib.pyplot.figure()
    matplotlib.pyplot.scatter(D0[0,:], D0[1,:], label="Spoofed")
    matplotlib.pyplot.scatter(D1[0,:], D1[1,:], label="Authentic")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig("%s/scatter_lda.png" % (output_folder))
    matplotlib.pyplot.close()
