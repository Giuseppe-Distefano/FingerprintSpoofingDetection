################################
#     MODULES IMPORTATIONS     #
################################
import modules.utility as utility
import modules.generative as gen
import modules.discriminative as dis
import modules.dataset as dataset
import modules.pca_lda as dr
import modules.svm as svm
import modules.gmm as gmm
import modules.calibration_fusion as calfus
import csv
import os
import numpy


############################
#     GLOBAL VARIABLES     #
############################
pi_t = 0.5
Cfn = 1
Cfp = 10
effective_prior = (pi_t*Cfn) / (pi_t*Cfn + (1-pi_t)*Cfp)
output_csv_name = "../output/Training/Results.csv"


#####################
#     FUNCTIONS     #
#####################
# ----- Setup folders and environment -----
def setup_environment ():
    if not os.path.exists('../output'): os.makedirs('../output')
    if not os.path.exists('../output/FeaturesAnalysis'): os.makedirs('../output/FeaturesAnalysis')
    if not os.path.exists('../output/FeaturesAnalysis/Histograms'): os.makedirs('../output/FeaturesAnalysis/Histograms')
    if not os.path.exists('../output/FeaturesAnalysis/Heatmaps'): os.makedirs('../output/FeaturesAnalysis/Heatmaps')
    if not os.path.exists('../output/DimensionalityReduction'): os.makedirs('../output/DimensionalityReduction')
    if not os.path.exists('../output/DimensionalityReduction/PCA'): os.makedirs('../output/DimensionalityReduction/PCA')
    if not os.path.exists('../output/DimensionalityReduction/LDA'): os.makedirs('../output/DimensionalityReduction/LDA')
    if not os.path.exists('../output/Training'): os.makedirs('../output/Training')
    if not os.path.exists('../output/Calibration_Fusion'): os.makedirs('../output/Calibration_Fusion')
    if not os.path.exists('../output/Evaluation'): os.makedirs('../output/Evaluation')


# ----- Load dataset -----
def load_dataset ():
    DTR, LTR = dataset.load_training_set()
    DTE, LTE = dataset.load_test_set()
    return (DTR,LTR), (DTE,LTE)


# ----- Analysis of features -----
def features_analysis (DTR, LTR):
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
    output_csv.write("Model,PCA,ZNorm,pi,lambda,C,gamma,G0,G1,Accuracy,Error rate,minDCF\n")
    output_csv.close()
    
    pca_values = [0, 9, 8, 7, 6, 5]         # value=0 when we don't apply PCA
    pi_values = [0.1, 0.5, 0.9, effective_prior]
    lambda_values = [1e-6, 1e-4, 1e-3, 1e-1, 1e+0, 1e+1, 1e+2]
    gmm_values = [2, 4, 8]
    C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e+0]
    gamma_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e+0]
    z_values = [0, 1]
    
    # pca_values = [0, 9]                     # value=0 when we don't apply PCA
    # pi_values = [0.1, 0.5]
    # lambda_values = [1e-6, 1e-1]
    # gmm_values = [2, 8]
    # C_values = [1e-2, 1e+0]
    # gamma_values = [1e-3, 1e-1]

    for pca_value in pca_values:
        for pi_value in pi_values:
            # Generative models
            gen.gen_kfold(D, L, K, pca_value, pi_value)

            # Discriminative models
            for z_value in z_values:
                for lambda_value in lambda_values:
                    dis.dis_kfold(D, L, K, pca_value, z_value, pi_value, lambda_value, z_value)
        
        # Support Vector Machines
        for C_value in C_values:
            for z_value in z_values:
                for gamma_value in gamma_values:
                    svm.svm_kfold(D, L, K, pca_value, z_value, C_value, gamma_value)
        
        # Gaussian Mixture Models
        if pca_value!=8 and pca_value!=6:
            for z_value in z_values:
                for g0_value in gmm_values:
                    for g1_value in gmm_values:
                        if g0_value>g1_value:
                            gmm.gmm_kfold(D, L, K, pca_value, z_value, g0_value, g1_value)


# ----- Sort training results with respect to minDCF -----
def sort_training_results ():
    # Sort results basing on minDCF
    with open("../output/Training/Results.csv", "r") as file:
        reader = csv.DictReader(file)
        next(reader)
        lines = list(reader)
    sorted_lines = sorted(lines, key=lambda x: float(x["minDCF"]), reverse=False)
    
    # Store ranking on a new CSV file
    fieldnames = ["Model", "PCA", "pi", "lambda", "C", "gamma", "G0", "G1", "Accuracy", "Error rate", "minDCF"]
    with open("../output/Training/SortedResults.csv", "w+") as file:
        file.write("Model,PCA,ZNorm,pi,lambda,C,gamma,G0,G1,Accuracy,Error rate,minDCF\n")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(sorted_lines)
    
    best_models = sorted_lines[:3]
    for model in best_models:
        print("%s \n" % model)

                        
# ----- Select the three best models and train them -----
def train_top3_models (D, L):
    train_model(D, L)
    sort_training_results()

    # ---- Quadratic Logistic Regression ----
    # --- PCA=8, ZNorm, pi=effective_prior, lambda=1e-2 ---
    qlr_scores = dis.train_qlr(D, L, 5, 8, 1, effective_prior, 1e-2)
    
    # ---- Polynomial Kernel SVM ----
    # --- No PCA, No ZNorm, C=1, gamma=1e-3 ---
    svm_scores = svm.train_pol_svm(D, L, 5, 0, 0, 1e+0, 1e-3)

    # ---- Diagonal Gaussian Mixture Model ----
    # --- No PCA, No ZNorm, G0=8, G1=2 ---
    gmm_scores = gmm.train_diagonal_gmm(D, L, 5, 0, 0, 8, 2)

    return qlr_scores, svm_scores, gmm_scores


# ----- Calibrate scores -----
def scores_calibration (L, qlr_scores, svm_scores, gmm_scores):
    qlr_scores = calfus.calibrate_scores(qlr_scores, L, 'QLR')
    svm_scores = numpy.hstack(numpy.array([]),numpy.array([]))#calfus.calibrate_scores(svm_scores, L, 'SVM')
    gmm_scores = numpy.hstack(numpy.array([]),numpy.array([]))#calfus.calibrate_scores(gmm_scores, L, 'GMM')

    return qlr_scores, svm_scores, gmm_scores


# ----- Fuse top3 models -----
def models_fusion (L, qlr_scores, svm_scores, gmm_scores):
    qlr_svm_scores = calfus.fuse_models('QLR', qlr_scores, 'SVM', svm_scores, L)
    qlr_gmm_scores = calfus.fuse_models('QLR', qlr_scores, 'GMM', gmm_scores, L)
    svm_gmm_scores = calfus.fuse_models('SVM', svm_scores, 'GMM', gmm_scores, L)
    
    return qlr_svm_scores, qlr_gmm_scores, svm_gmm_scores


# ----- Evaluate model -----
def model_evaluation (DTR, LTR, DTE, LTE):
    pass


###############################
#     MAIN OF THE PROGRAM     #
###############################
if __name__ == "__main__":
    # ---- Setup environment ----
    setup_environment()

    # ---- Dataset analysis ----
    (DTR,LTR), (DTE,LTE) = load_dataset()
    DTR,LTR = utility.randomize(DTR, LTR, 0)
    features_analysis(DTR, LTR)
    dimensionality_reduction(DTR, LTR)

    # ---- Training ----
    qlr_scores, svm_scores, gmm_scores = train_top3_models(DTR, LTR)

    # ---- Scores calibration ----
    qlr_scores_cal, svm_scores_cal, gmm_scores_cal = scores_calibration(LTR, qlr_scores, svm_scores, gmm_scores)
    
    # ---- Models fusion ----
    qlr_svm_scores, qlr_gmm_scores, svm_gmm_scores = models_fusion(LTR, qlr_scores_cal, svm_scores_cal, gmm_scores_cal)

    # ---- Evaluation ----
    model_evaluation(DTR, LTR, DTE, LTE)
