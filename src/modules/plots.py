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
# ----- Plot dataset histograms -----
def plot_dataset_histograms (D, L):
    output_folder = "../output/00_histograms"
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    for x in range(dataset.features):
        plt.figure()
        plt.title("Feature %d" % x)
        plt.hist(D0[x,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Spoofed")
        plt.hist(D1[x,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Authentic")
        plt.legend()
        plt.savefig("%s/histogram_%d.png" % (output_folder, x))
        plt.close()


# ----- Plot dataset heatmaps -----
def plot_dataset_heatmaps (D, L):
    output_folder = "../output/01_heatmaps"

    # All samples
    corr = np.zeros((dataset.features, dataset.features))
    for x in range(dataset.features):
        for y in range(dataset.features):
            corr[x][y] = utility.compute_correlation(D[x,:], D[y,:])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="YlGnBu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_all.png" % (output_folder))

    # Only spoofed fingerprint samples
    corr = np.zeros((dataset.features, dataset.features))
    for x in range(dataset.features):
        for y in range(dataset.features):
            corr[x][y] = utility.compute_correlation(D[x,L==0], D[y,L==0])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="coolwarm", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_spoofed.png" % (output_folder))

    # Only authentic fingerprint samples
    corr = np.zeros((dataset.features, dataset.features))
    for x in range(dataset.features):
        for y in range(dataset.features):
            corr[x][y] = utility.compute_correlation(D[x,L==1], D[y,L==1])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="BuPu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_authentic.png" % (output_folder))


# ----- Plot histograms -----
def plot_histograms (D, L, distinct_classes, bins, output_folder):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    for x in range(distinct_classes):
        plt.figure()
        plt.hist(D0[x,:], bins=bins, density=True, alpha=0.4, edgecolor="black", label="Spoofed")
        plt.hist(D1[x,:], bins=bins, density=True, alpha=0.4, edgecolor="black", label="Authentic")
        plt.legend()
        plt.savefig("%s/histogram_%d.png" % (output_folder, x))
        plt.close()


##### Plot LDA histograms #####
def plot_lda_histograms (D, L, bins, output_folder):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    plt.figure()
    plt.hist(D0[0,:], bins=bins, density=True, alpha=0.4, edgecolor="black", label="Spoofed")
    plt.hist(D1[0,:], bins=bins, density=True, alpha=0.4, edgecolor="black", label="Authentic")
    plt.legend()
    plt.savefig("%s/lda_histogram.png" % (output_folder))
    plt.close()


##### Plot heatmaps #####
def plot_heatmaps (D, L, distinct_classes, output_folder):
    # All samples
    corr = np.zeros((distinct_classes, distinct_classes))
    for x in range(distinct_classes):
        for y in range(distinct_classes):
            corr[x][y] = utility.compute_correlation(D[x,:], D[y,:])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="YlGnBu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_all.png" % (output_folder))
    
    # Only samples labelled with 0 (spoofed fingerprint samples)
    corr = np.zeros((distinct_classes, distinct_classes))
    for x in range(distinct_classes):
        for y in range(distinct_classes):
            corr[x][y] = utility.compute_correlation(D[x,L==0], D[y,L==0])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="coolwarm", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_spoofed.png" % (output_folder))
    
    # Only samples labelled with 1 (authentic fingerprint samples)
    corr = np.zeros((distinct_classes, distinct_classes))
    for x in range(distinct_classes):
        for y in range(distinct_classes):
            corr[x][y] = utility.compute_correlation(D[x,L==1], D[y,L==1])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="BuPu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_authentic.png" % (output_folder))


##### Plot scatters #####
def plot_scatters (D, L, distinct_classes, output_folder):
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    for x1 in range(distinct_classes):
        for x2 in range(distinct_classes):
            if (x1>=x2): continue
            plt.figure()
            plt.scatter(D0[x1,:], D0[x2,:], label="Spoofed")
            plt.scatter(D1[x1,:], D1[x2,:], label="Authentic")
            plt.legend()
            plt.savefig("%s/scatter_%d_%d.png" % (output_folder, x1, x2))
            plt.close()