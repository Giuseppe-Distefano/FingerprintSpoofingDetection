################################
##### MODULES IMPORTATIONS #####
################################
import modules.utility as utility
import modules.generative as gen
import modules.discriminative as dis
import modules.dataset as dataset
import modules.pca_lda as dr
import modules.svm as svm
import modules.gmm as gmm


####################
# GLOBAL VARIABLES #
####################
pi_t = 0.5
Cfn = 1
Cfp = 10
effective_prior = (pi_t*Cfn) / (pi_t*Cfn + (1-pi_t)*Cfp)
output_csv_name = "../output/Training/Results.csv"


#####################
##### FUNCTIONS #####
#####################
# ----- Load dataset -----
def load_dataset ():
    DTR, LTR = dataset.load_training_set()
    DTE, LTE = dataset.load_test_set()
    return (DTR,LTR), (DTE,LTE)


# ----- Analysis of features -----
def features__analysis (DTR, LTR):
    dataset.plot_histograms(DTR, LTR)
    dataset.plot_heatmaps(DTR, LTR)


# ----- Dimensionality reduction -----
def dimensionality_reduction (D, L):
    m = 2
    dr.apply_pca(D, L, m, "../output/DimensionalityReduction/PCA")
    dr.apply_lda(D, L, m, "../output/DimensionalityReduction/LDA")


# ----- Train model -----
def train_model (D, L):
    K = 5

    output_csv = open(output_csv_name, "w")
    output_csv.write("Model,PCA,pi,lambda,C,gamma,G0,G1,Accuracy,Error rate,minDCF")
    output_csv.close()
    
    pca_values = [0, 9, 8, 7, 6, 5]         # value=0 when we don't apply PCA
    pi_values = [0.1, 0.5, 0.9, effective_prior]
    lambda_values = [1e-6, 1e-4, 1e-3, 1e-1, 1e+0, 1e+1, 1e+2]
    gmm_values = [1, 2, 4, 8]
    C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e+0]
    gamma_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e+0]
    
    pca_values = [0, 9]                     # value=0 when we don't apply PCA
    pi_values = [0.1, 0.5]
    lambda_values = [1e-6, 1e-1]
    gmm_values = [2, 8]
    C_values = [1e-2, 1e+0]
    gamma_values = [1e-3, 1e-1]

    for pca_value in pca_values:
        for pi_value in pi_values:
            # Generative models
            gen.gen_kfold(D, L, K, pca_value, pi_value)

            # Discriminative models
            for lambda_value in lambda_values:
                dis.dis_kfold(D, L, K, pca_value, pi_value, lambda_value)
        
        # Support Vector Machines
        for C_value in C_values:
            for gamma_value in gamma_values:
                svm.svm_kfold(D, L, K, pca_value, C_value, gamma_value)
        
        # Gaussian Mixture Models
        for g0_value in gmm_values:
            for g1_value in gmm_values:
                gmm.gmm_kfold(D, L, K, pca_value, g0_value, g1_value)


###############################
##### MAIN OF THE PROGRAM #####
###############################
if __name__ == "__main__":
    (DTR,LTR), (DTE,LTE) = load_dataset()    
    features__analysis(DTR, LTR)
    dimensionality_reduction(DTR, LTR)
    train_model(DTR, LTR)
