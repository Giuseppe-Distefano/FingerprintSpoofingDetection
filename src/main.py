################################
##### MODULES IMPORTATIONS #####
################################
import modules.utility as utility
import modules.plots as plots
import modules.generative as gen
import modules.discriminative as dis
import modules.dataset as dataset
import modules.pca_lda as dr
import numpy as np


#####################
##### FUNCTIONS #####
#####################
# ----- Load dataset -----
def load_dataset ():
    DTR, LTR = dataset.load_training_set("../data/Train.txt")
    DTE, LTE = dataset.load_test_set("../data/Test.txt")
    return (DTR,LTR), (DTE,LTE)


# ----- Analysis of features -----
def features__analysis (DTR, LTR):
    plots.plot_dataset_histograms(DTR, LTR, "../output/00_histograms")
    plots.plot_dataset_heatmaps(DTR, LTR, "../output/01_heatmaps")


# ----- Dimensionality reduction -----
def dimensionality_reduction (D, L):
    m = 2
    dr.apply_pca(D, L, m, "../output/02_PCA")
    dr.apply_lda(D, L, m, "../output/03_LDA")


# ----- Train model -----
def train_model (D, L):
    K = 5
    pca_values = [0]#pca_values = [0, 9, 8, 7, 6, 5] # value=0 when we don't apply PCA
    pi_values = [0.5]#pi_values = [0.1, 0.5, 0.9]
    for i,pca_value in enumerate(pca_values):
        for j,pi_value in enumerate(pi_values):
            output1 = "../output/04_Kfold/performance_" + str(i) + "_" + str(j) + ".txt"
            output2 = "../output/05_DCF/performance_" + str(i) + "_" + str(j) + ".txt"
            # Generative models
            #gen.kfold(D, L, K, pca_value, pi_value, output1, output2)
            # Discriminative models
            dis.kfold(D, L, K, pca_value, pi_value, output1, output2)


###############################
##### MAIN OF THE PROGRAM #####
###############################
if __name__ == "__main__":
    (DTR,LTR), (DTE,LTE) = load_dataset()    
    #features__analysis(DTR, LTR)
    #dimensionality_reduction(DTR, LTR)
    train_model(DTR, LTR)
