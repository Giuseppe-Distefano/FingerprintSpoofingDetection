##### LIBRARIES #####
import numpy
import utils as utils
import discriminative as discriminative
import generative as generative
import gmm as gmm
import svm as svm
import plots as plots
import calibration as calibration
import costs as costs
import tuning

##### GLOBAL VARIABLES #####
pi_eff = 0.5
Cfn = Cfn_eff = 1
Cfp = Cfp_eff = 10
effective_prior = (pi_eff*Cfn_eff) / (pi_eff*Cfn_eff + (1-pi_eff)*Cfp_eff)

pca_values = [0, 9, 8, 7, 6, 5]
pi_values = [0.1, 0.5, 0.9]
lambda_values = [1e-6, 1e-4, 1e-3, 1e-1, 1e+0, 1e+1, 1e+2]
z_values = [0, 1]
g_values = [2, 4, 8]


##### FUNCTIONS #####
# ----- Training -----
def train(DTR_RAND, LTR_RAND):
    # --- Generative models ---
    for pca_value in pca_values:
        generative.mvg_train(DTR_RAND, LTR_RAND, 5, pca_value)
        generative.naive_bayes_tied_train(DTR_RAND, LTR_RAND, 5, pca_value)
        generative.naive_bayes_train(DTR_RAND, LTR_RAND, 5, pca_value)
        generative.mvg_tied_train(DTR_RAND, LTR_RAND, 5, pca_value)
    
    # --- Discriminative models ---
    for pca_value in pca_values:
        for pi_value in pi_values:
            for lambda_value in lambda_values:
                for z_value in z_values:
                    discriminative.linear_lr_train(DTR_RAND, LTR_RAND, 5, z_value, lambda_value, pi_value, pca_value)
                    discriminative.quadratic_lr_train(DTR_RAND, LTR_RAND, 5, z_value, lambda_value, pi_value, pca_value)
    
    # --- SVM models ---
    for pca_value in pca_values:
        for z_value in z_values:
            svm.SVM_linear_training(DTR_RAND, LTR_RAND, 5, 1, 1e-3, z_value, pca_value)
            svm.SVM_kernel_quadratic_training(DTR_RAND, LTR_RAND, 5, 1, 1e-3, z_value, pca_value)
            svm.SVM_RBF_quadratic_training(DTR_RAND, LTR_RAND, 5, 1, 1e-3, z_value, pca_value)

    # --- GMM models ---
    for pca_value in pca_values:
        if pca_value in [0, 8, 5]:
            for g0_value in g_values:
                for g1_value in g_values:
                    if g0_value>g1_value:
                        for z_value in z_values:
                            gmm.GMM_train(DTR_RAND, LTR_RAND, 5, g0_value, g1_value, z_value, False, False, pca_value)#Full
                            gmm.GMM_train(DTR_RAND, LTR_RAND, 5, g0_value, g1_value, z_value, False, True, pca_value)#Tied
                            gmm.GMM_train(DTR_RAND, LTR_RAND, 5, g0_value, g1_value, z_value, True, False, pca_value)#Diagonal
                            gmm.GMM_train(DTR_RAND, LTR_RAND, 5, g0_value, g1_value, z_value, True, True, pca_value)#Tied Diagonal

# ----- Scores calibration and fusion -----
def calibration_fusion (DTR_RAND, LTR_RAND):
    # --- Calibration ---
        qlr_scores = discriminative.quadratic_lr_train(DTR_RAND, LTR_RAND, 5, 1, 1e-2, effective_prior, 8)
        qlr_cal,_,l = calibration.calibrate_scores(qlr_scores, LTR_RAND)
        plots.bayesErrorPlot(numpy.hstack(qlr_scores), LTR_RAND, 'Uncalibrated QLR')
        plots.bayesErrorPlot(numpy.hstack(qlr_cal), l, 'Calibrated QLR')

        svm_scores = svm.SVM_kernel_quadratic_training(DTR_RAND, LTR_RAND, 5, 1, 1e-3, 0, 8)
        svm_cal,_,l = calibration.calibrate_scores(svm_scores, LTR_RAND)
        plots.bayesErrorPlot(numpy.hstack(svm_scores), LTR_RAND, 'Uncalibrated SVM')
        plots.bayesErrorPlot(numpy.hstack(svm_cal), l, 'Calibrated SVM')

        gmm_scores = gmm.GMM_train(DTR_RAND, LTR_RAND, 5, 8, 2, 0, True, False, 0)
        gmm_cal,_,l = calibration.calibrate_scores(gmm_scores, LTR_RAND)
        plots.bayesErrorPlot(numpy.hstack(gmm_scores), LTR_RAND, 'Uncalibrated GMM')
        plots.bayesErrorPlot(numpy.hstack(gmm_cal), l, 'Calibrated GMM')


        # --- Fusion ---
        qlr_svm_scores,_,lf = calibration.fuse_scores(numpy.hstack(qlr_cal), numpy.hstack(svm_cal), l)
        plots.bayesErrorPlot(numpy.hstack(qlr_svm_scores), lf, 'Fusion of QLR and SVM')

        gmm_svm_scores,_,lf = calibration.fuse_scores(numpy.hstack(gmm_cal), numpy.hstack(svm_cal), l)
        plots.bayesErrorPlot(numpy.hstack(gmm_svm_scores), lf, 'Fusion of GMM and SVM')
        
        gmm_qlr_scores,_,lf = calibration.fuse_scores(numpy.hstack(gmm_cal), numpy.hstack(qlr_cal), l)
        plots.bayesErrorPlot(numpy.hstack(gmm_qlr_scores), lf, 'Fusion of GMM and QLR')

        return ((qlr_scores, svm_scores, gmm_scores), (qlr_cal, svm_cal, gmm_cal), (qlr_svm_scores, gmm_qlr_scores, gmm_svm_scores))
        

# ----- Evaluation -----
def evaluation (DTR_RAND, LTR_RAND, DTE, LTE):
    (qlr_scores, svm_scores) = calibration_fusion(DTR_RAND, LTR_RAND)

    #(qlr_scores, svm_scores, gmm_scores), (qlr_cal, svm_cal, gmm_cal), (qlr_svm_scores, gmm_qlr_scores, gmm_svm_scores) = calibration_fusion(DTR_RAND, LTR_RAND)
    svm_scores = numpy.hstack(svm_scores)
    svm_scores_e = numpy.hstack(svm.SVM_kernel_quadratic_evaluation(D, L, DTE, LTE, 0, 8))
    svm_scores_c = discriminative.linear_lr_eval(utils.vrow(svm_scores), LTR_RAND, utils.vrow(svm_scores_e), LTE, 0, 1e-3, pi_eff, 0)
    plots.plotDCF_lambda_eval(svm_scores_c, LTE, "SVM evaluation")
    
    gmm_scores = numpy.hstack(gmm_scores)
    gmm_scores_e = numpy.hstack(gmm.GMM_eval(D, L, DTE, LTE, 8, 2, 0, True, False, 0))
    gmm_scores_c = discriminative.linear_lr_eval(utils.vrow(gmm_scores), LTR_RAND, utils.vrow(gmm_scores_e), LTE, 0, 1e-3, pi_eff, 0)

    plots.bayesErrorPlot(gmm_scores_c, LTE, "GMM evaluation")

    qlr_scores = numpy.hstack(qlr_scores)
    qlr_scores_e = numpy.hstack(discriminative.quadratic_lr_evaluation(D, L, DTE, LTE, 1, 1e-2, effective_prior, 8))
    qlr_scores_c = discriminative.linear_lr_eval(utils.vrow(qlr_scores), LTR_RAND, utils.vrow(qlr_scores_e), LTE, 0, 1e-3, pi_eff, 0)
    plots.bayesErrorPlot(qlr_scores_c, LTE, "QLR evaluation")



    gmm_svm_t = numpy.vstack([gmm_scores, svm_scores])
    gmm_svm_e = numpy.vstack([gmm_scores_c, svm_scores_c])
    gmm_svm_c = discriminative.linear_lr_eval(gmm_svm_t, LTR_RAND, gmm_svm_e, LTE, 0, 1e-3, pi_eff, 0)
    plots.bayesErrorPlot(gmm_svm_c, LTE, "GMM-SVM evaluation")

    gmm_qlr_t = numpy.vstack([gmm_scores, qlr_scores])
    gmm_qlr_e = numpy.vstack([gmm_scores_c, qlr_scores_c])
    gmm_qlr_c = discriminative.linear_lr_eval(gmm_qlr_t, LTR_RAND, gmm_qlr_e, LTE, 0, 1e-3, pi_eff, 0)
    plots.bayesErrorPlot(gmm_qlr_c, LTE, "GMM-QLR evaluation")

    qlr_svm_t = numpy.vstack([qlr_scores, svm_scores])
    qlr_svm_e = numpy.vstack([qlr_scores_c, svm_scores_c])
    qlr_svm_c = discriminative.linear_lr_eval(qlr_svm_t, LTR_RAND, qlr_svm_e, LTE, 0, 1e-3, pi_eff, 0)
    plots.bayesErrorPlot(qlr_svm_c, LTE, "QLR-SVM evaluation")

# ----- Tuning -----
def hyperparameter_tuning(D,L):
    # LR
    tuning.lambda_estimation(D,L)
    tuning.lambda_estimation_quadratic(D,L)

    # SVM
    tuning.C_estimation_linear(DTR_RAND, LTR_RAND, "SVM LINEAR (K = 1)")
    tuning.C_estimation_linear(DTR_RAND, LTR_RAND, "SVM LINEAR (K = 10)")

    tuning.c_estimation_poly(DTR_RAND,LTR_RAND,0, "SVM Polynomial c = 0") 
    tuning.c_estimation_poly(DTR_RAND,LTR_RAND,1, "SVM Polynomial c = 1") 

    tuning.c_gamma_estimation_rbf(DTR_RAND,LTR_RAND,False,False, "RAW")
    tuning.c_gamma_estimation_rbf(DTR_RAND,LTR_RAND,True,False, "PCA (No Z NORM)")
    tuning.c_gamma_estimation_rbf(DTR_RAND,LTR_RAND,False,True, "Z NORM (No PCA)")
    tuning.c_gamma_estimation_rbf(DTR_RAND,LTR_RAND,True,True, "PCA Z NORM")

    #GMM
    tuning.estimation_gmm_components(DTR_RAND, LTR_RAND, False, False,False,False, "GMM Full Covariance")
    tuning.estimation_gmm_components(DTR_RAND, LTR_RAND, False, True,False,False, "GMM Tied Covariance")
    tuning.estimation_gmm_components(DTR_RAND, LTR_RAND, True, False,False,False, "GMM Diagonal Covariance")
    tuning.estimation_gmm_components(DTR_RAND, LTR_RAND, True, True,False,False ,"GMM Tied Diagonal Covariance")

    #tuning.estimation_gmm_components_eval(D,L,DTE,LTE, True, False, False, False, "GMM Diagonal Evaluation")
    #tuning.estimation_gmm_components_eval(D,L,DTE,LTE, False, True, False, False, "GMM Tied Evaluation")
    #tuning.estimation_gmm_components_eval(D,L,DTE,LTE, False, False, False, False, "GMM Full Evaluation")
    #tuning.estimation_gmm_components_eval(D,L,DTE,LTE, True, False, False, False, "GMM Diagonal PCA Evaluation")
    #tuning.estimation_gmm_components_eval(D,L,DTE,LTE, True, True, False, False, "GMM Tied Diagonal Evaluation")
    



if __name__=="__main__":
    D, L = utils.load_data(".\src\Train.txt")
    DTE, LTE = utils.load_data(".\src\Test.txt")
    DTR_RAND, LTR_RAND = utils.randomize(D,L,0)

    hyperparameter_tuning(DTR_RAND,LTR_RAND)
    train(DTR_RAND, LTR_RAND)
    calibration_fusion(DTR_RAND, LTR_RAND)
    evaluation(DTR_RAND,LTR_RAND, DTE, LTE)


