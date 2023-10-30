################################
#     MODULES IMPORTATIONS     #
################################
import modules.utility as utility
import numpy as np
import scipy.optimize as sopt
import modules.pca_lda as dr
import modules.costs as dcf
import scipy.special as sspec


####################
#     FUNCTIONS    #
####################
# ----- Naive Bayes without tied covariance -----
def naive_bayes (DTR, LTR, DTE, LTE):
    # Compute mean and covariance for each class
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0, cov0 = utility.row_to_column(utility.compute_mean(D0)), utility.compute_covariance(D0)*np.identity(DTR.shape[0])
    mu1, cov1 = utility.row_to_column(utility.compute_mean(D1)), utility.compute_covariance(D1)*np.identity(DTR.shape[0])

    # Compute likelihoods
    S = np.empty((2,DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = np.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu0, cov0))
        S[1,i] = np.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu1, cov1))

    # Compute log-likelihood ratio and use it to classify samples
    llr = np.log(S[1,:] / S[0,:])
    #predicted = utility.predict_labels(llr, 0)
    #wrong_predictions = utility.count_mispredictions(predicted, LTE)
    predicted_labels = np.log(S).argmax(axis=0)
    wrong_predictions = len(LTE) - np.array(predicted_labels == LTE).sum()

    return wrong_predictions, llr


# ----- Naive Bayes with tied covariance -----
def tied_naive_bayes (DTR, LTR, DTE, LTE):
    # Compute mean for each class and tied covariance matrix
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0, cov0 = utility.row_to_column(utility.compute_mean(D0)), utility.compute_covariance(D0)*np.identity(DTR.shape[0])
    mu1, cov1 = utility.row_to_column(utility.compute_mean(D1)), utility.compute_covariance(D1)*np.identity(DTR.shape[0])
    tied_cov = (cov0*D0.shape[1] + cov1*D1.shape[1]) / DTR.shape[1]

    # Compute likelihoods
    S = np.empty((2,DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = np.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu0, tied_cov))
        S[1,i] = np.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu1, tied_cov))
    
    # Compute log-likelihood ratio and use it to classify samples
    llr = np.log(S[1,:] / S[0,:])
    #predicted = utility.predict_labels(llr, 0)
    #wrong_predictions = utility.count_mispredictions(predicted, LTE)
    predicted_labels = np.log(S).argmax(axis=0)
    wrong_predictions = len(LTE) - np.array(predicted_labels == LTE).sum()

    return wrong_predictions, llr


# ----- Multivariate Gaussian without tied covariance -----
def mvg (DTR, LTR, DTE, LTE):
    # Compute mean and covariance for each class
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0, cov0 = utility.row_to_column(utility.compute_mean(D0)), utility.compute_covariance(D0)
    mu1, cov1 = utility.row_to_column(utility.compute_mean(D1)), utility.compute_covariance(D1)

    # Compute likelihoods
    S = np.empty((2,DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = np.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu0, cov0))
        S[1,i] = np.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu1, cov1))
    
    # Compute log-likelihood ratio and use it to classify samples
    llr = np.log(S[1,:] / S[0,:])
    #predicted = utility.predict_labels(llr, 0)
    #wrong_predictions = utility.count_mispredictions(predicted, LTE)
    predicted_labels = np.log(S).argmax(axis=0)
    wrong_predictions = len(LTE) - np.array(predicted_labels == LTE).sum()

    return wrong_predictions, llr


# ----- Multivariate Gaussian with tied covariance -----
def tied_mvg (DTR, LTR, DTE, LTE):
    # Compute mean for each class and tied covariance matrix
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0, cov0 = utility.row_to_column(utility.compute_mean(D0)), utility.compute_covariance(D0)
    mu1, cov1 = utility.row_to_column(utility.compute_mean(D1)), utility.compute_covariance(D1)
    tied_cov = (cov0*D0.shape[1] + cov1*D1.shape[1]) / DTR.shape[1]

    # Compute likelihoods
    S = np.empty((2,DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = np.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu0, tied_cov))
        S[1,i] = np.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu1, tied_cov))
    
    # Compute log-likelihood ratio and use it to classify samples
    llr = np.log(S[1,:] / S[0,:])
    #predicted = utility.predict_labels(llr, 0)
    #wrong_predictions = np.array(predicted!=LTE).sum()
    predicted_labels = np.log(S).argmax(axis=0)
    wrong_predictions = len(LTE) - np.array(predicted_labels == LTE).sum()

    return wrong_predictions, llr


# ----- Compare classifiers using K-Fold -----
def kfold (D, L, K, pca_value, pi_value, output_filename1, output_filename2):
    classifiers = [
        (mvg, "MVG"),
        (tied_mvg, "MVG with tied covariance"),
        (naive_bayes, "Naive Bayes"),
        (tied_naive_bayes, "Naive Bayes with tied covariance")
    ]
    output_file1 = open(output_filename1, "w+")
    output_file2 = open(output_filename2, "w+")
    Cfn = 1
    Cfp = 10
    N = int(D.shape[1]/K)
    
    for j,(fun,name) in enumerate(classifiers):
        wrong_predictions = 0
        np.random.seed(j)
        ll_ratios = []
        labels = []
        indexes = np.random.permutation(D.shape[1])

        print("%s \t PCA %.3f \t pi_t %.3f \n" % (name, pca_value, pi_value))

        for i in range(K):
            # Select which subset to use for evaluation
            idxTest = indexes[i*N:(i+1)*N]
            if i>0: idxTrainLeft = indexes[0:i*N]
            elif (i+1)<K: idxTrainRight = indexes[(i+1)*N:]
            if i==0: idxTrain = idxTrainRight
            elif (i+1)==K: idxTrain = idxTrainLeft
            else: idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
            DTR = D[:,idxTrain]
            LTR = L[idxTrain]
            DTE = D[:,idxTest]
            LTE = L[idxTest]

            # Apply PCA if necessary
            if pca_value!=0:
                P = dr.apply_pca(DTR, LTR, pca_value)
                DTR = np.dot(P.T, DTR)
                DTE = np.dot(P.T, DTE)

            # Apply classifier
            wrong, llr = fun(DTR, LTR, DTE, LTE)
            wrong_predictions += wrong
            #ll_ratios.append(utility.row_to_column(llr))
            ll_ratios.append(llr)
            #labels.append(utility.row_to_column(LTE))
            labels.append(LTE)

        # Evaluate accuracy and error rate
        error_rate = wrong_predictions / D.shape[1]
        accuracy = 1 - error_rate
        output_file1.write("%s\n" % (name))
        output_file1.write("  Accuracy: %.3f%%\n" % (100.0*accuracy))
        output_file1.write("  Error rate: %.3f%%\n" % (100.0*error_rate))
        output_file1.write("\n")

        # Compute min DCF
        cost = dcf.compute_min_DCF(pi_value, Cfn, Cfp, np.hstack(ll_ratios), np.hstack(labels))
        output_file2.write("%s\n" % (name))
        output_file2.write("  PCA: %d, pi: %.3f\n" % (pca_value, pi_value))
        output_file2.write("  min DCF: %.3f\n" % (cost))
        output_file2.write("\n")

        print("  Accuracy %.3f \t Error rate %.3f \t minDCF %.5f \n" % ((100.0*accuracy), (100.0*error_rate), cost))