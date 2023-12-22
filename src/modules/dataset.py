################################
#     MODULES IMPORTATIONS     #
################################
import matplotlib.pyplot
import numpy
import seaborn
import modules.utility as utility


############################
#     GLOBAL VARIABLES     #
############################
features = int(10)
distinct_classes = int(2)
training_input = "../data/Train.txt"
test_input = "../data/Test.txt"
histograms_folder = "../output/FeaturesAnalysis/Histograms"
heatmaps_folder = "../output/FeaturesAnalysis/Heatmaps"


#####################
#     FUNCTIONS     #
#####################
# ----- Read file -----
def read_file (filename):
    D = []
    L = []
    with open(filename) as file:
        for line in file:
            try:
                attributes = line.split(",")[0:features]
                attributes = utility.row_to_column(numpy.array([float(i) for i in attributes]))
                label = int(line.split(",")[-1].strip())
                D.append(attributes)
                L.append(label)
            except:
                pass
    return numpy.hstack(D), numpy.array(L, dtype=numpy.int32)


# ----- Load training set -----
def load_training_set ():
    DTR,LTR = read_file(training_input)
    return DTR,LTR


# ----- Load test set -----
def load_test_set ():
    DTE,LTE = read_file(test_input)
    return DTE,LTE


# ----- Plot histograms of training set -----
def plot_histograms (DTR, LTR):
    # Split according to label
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]

    # Plot histograms for each feature
    for x in range(features):
        matplotlib.pyplot.figure()
        matplotlib.pyplot.title("Feature %d" % x)
        matplotlib.pyplot.hist(D0[x,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Spoofed")
        matplotlib.pyplot.hist(D1[x,:], bins=40, density=True, alpha=0.4, edgecolor="black", label="Authentic")
        matplotlib.pyplot.legend()
        matplotlib.pyplot.savefig("%s/histogram_%d.png" % (histograms_folder, x))
        matplotlib.pyplot.close()


# ----- Plot heatmaps of training set ---
def plot_heatmaps (DTR, LTR):
    # Consider all samples
    corr = numpy.zeros((features, features))
    for x in range(features):
        for y in range(features):
            corr[x][y] = utility.compute_correlation(DTR[x,:], DTR[y,:])
    seaborn.set()
    heatmap = seaborn.heatmap(numpy.abs(corr), cmap="YlGnBu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_all.png" % (heatmaps_folder))

    # Consider only samples labeled as spoofed fingerprints
    corr = numpy.zeros((features, features))
    for x in range(features):
        for y in range(features):
            corr[x][y] = utility.compute_correlation(DTR[x,LTR==0], DTR[y,LTR==0])
    seaborn.set()
    heatmap = seaborn.heatmap(numpy.abs(corr), cmap="coolwarm", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_spoofed.png" % (heatmaps_folder))

    # Consider only samples labeled as authentic fingerprints
    corr = numpy.zeros((features, features))
    for x in range(features):
        for y in range(features):
            corr[x][y] = utility.compute_correlation(DTR[x,LTR==1], DTR[y,LTR==1])
    seaborn.set()
    heatmap = seaborn.heatmap(numpy.abs(corr), cmap="BuPu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_authentic.png" % (heatmaps_folder))
