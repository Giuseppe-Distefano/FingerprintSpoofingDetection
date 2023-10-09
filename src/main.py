################################
##### MODULES IMPORTATIONS #####
################################
import modules.utility as utility
import modules.plots as plots
import modules.train as train
import modules.dataset as dataset
import modules.pca_lda as dr


#####################
##### FUNCTIONS #####
#####################
# ----- Load dataset -----
def load_dataset ():
    DTR, LTR = dataset.load_training_set()
    DTE, LTE = dataset.load_test_set()
    return (DTR,LTR), (DTE,LTE)


# ----- Analysis of features -----
def features__analysis (D, L):
    plots.plot_dataset_histograms(D, L)
    plots.plot_dataset_heatmaps(D, L)


# ----- Dimensionality reduction -----
def dimensionality_reduction (D, L):
    m = 2
    dr.apply_pca(D, L, m)
    dr.apply_lda(D, L, m)


# ----- K fold -----
def Kfold (D, L):
    K = 5
    train.kfold(D, L, K)


###############################
##### MAIN OF THE PROGRAM #####
###############################
if __name__ == "__main__":
    (DTR,LTR), (DTE,LTE) = load_dataset()    
    #features__analysis(DTR, LTR)
    #dimensionality_reduction(DTR, LTR)
    Kfold(DTR, LTR)
