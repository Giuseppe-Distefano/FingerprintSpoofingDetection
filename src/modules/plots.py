################################
#     MODULES IMPORTATIONS     #
################################
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import modules.utility as utility
import modules.dataset as dataset


#####################
#     FUNCTIONS     #
#####################
# ----- Plot histograms of training set -----
def plot_dataset_histograms (DTR, LTR, output_folder):
    # Split according to label
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]

    # Plot histograms for each feature
    for x in range(dataset.features):
        plt.figure()
        plt.title("Feature %d" % x)
        plt.hist(D0[x,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Spoofed")
        plt.hist(D1[x,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Authentic")
        plt.legend()
        plt.savefig("%s/histogram_%d.png" % (output_folder, x))
        plt.close()


# ----- Plot heatmaps of training set ---
def plot_dataset_heatmaps (DTR, LTR, output_folder):
    # Consider all samples
    corr = np.zeros((dataset.features, dataset.features))
    for x in range(dataset.features):
        for y in range(dataset.features):
            corr[x][y] = utility.compute_correlation(DTR[x,:], DTR[y,:])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="YlGnBu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_all.png" % (output_folder))

    # Consider only samples labeled as spoofed fingerprints
    corr = np.zeros((dataset.features, dataset.features))
    for x in range(dataset.features):
        for y in range(dataset.features):
            corr[x][y] = utility.compute_correlation(DTR[x,LTR==0], DTR[y,LTR==0])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="coolwarm", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_spoofed.png" % (output_folder))

    # Consider only samples labeled as authentic fingerprints
    corr = np.zeros((dataset.features, dataset.features))
    for x in range(dataset.features):
        for y in range(dataset.features):
            corr[x][y] = utility.compute_correlation(DTR[x,LTR==1], DTR[y,LTR==1])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="BuPu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_authentic.png" % (output_folder))


# ----- Plot scatters of data projected after PCA -----
def plot_pca_scatters (D, L, output_folder):
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    plt.figure()
    plt.scatter(D0[0,:], D0[1,:], label="Spoofed")
    plt.scatter(D1[0,:], D1[1,:], label="Authentic")
    plt.legend()
    plt.savefig("%s/scatter_pca.png" % (output_folder))
    plt.close()


# ----- Plot histograms of data projected after LDA -----
def plot_lda_histograms (D, L, output_folder):
    # Split according to label
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    # Plot histograms for each feature
    plt.figure()
    plt.hist(D0[0,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Spoofed")
    plt.hist(D1[0,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Authentic")
    plt.legend()
    plt.savefig("%s/histogram_lda.png" % (output_folder))
    plt.close()


# ----- Plot scatters of data projected after LDA -----
def plot_lda_scatters (D, L, output_folder):
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    plt.figure()
    plt.scatter(D0[0,:], D0[1,:], label="Spoofed")
    plt.scatter(D1[0,:], D1[1,:], label="Authentic")
    plt.legend()
    plt.savefig("%s/scatter_lda.png" % (output_folder))
    plt.close()
    