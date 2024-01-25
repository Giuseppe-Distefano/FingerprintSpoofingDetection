################################
#     MODULES IMPORTATIONS     #
################################
import numpy
import modules.utility as utility
import modules.pca_lda as dr
import modules.costs as dcf


####################
# GLOBAL VARIABLES #
####################
generative_training_output = "../output/Training/Generative.txt"
output_csv_name = "../output/Training/Results.csv"
Cfn = 1
Cfp = 10


####################
#     FUNCTIONS    #
####################
# ----- Naive Bayes without tied covariance -----
def naive_bayes (DTR, LTR, DTE, LTE):
    # Compute mean and covariance for each class
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0, cov0 = utility.compute_mean(D0), utility.compute_covariance(D0)*numpy.identity(DTR.shape[0])
    mu1, cov1 = utility.compute_mean(D1), utility.compute_covariance(D1)*numpy.identity(DTR.shape[0])

    # Compute likelihoods
    S = numpy.empty((2,DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = numpy.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu0, cov0))
        S[1,i] = numpy.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu1, cov1))

    # Compute log-likelihood ratio and use it to classify samples
    llr = numpy.log(S[1,:] / S[0,:])
    # predicted_labels = numpy.log(S).argmax(axis=0)
    # wrong_predictions = len(LTE) - numpy.array(predicted_labels == LTE).sum()
    predicted_labels = utility.predict_labels(llr, 0)
    wrong_predictions = utility.count_mispredictions(predicted_labels, LTE)

    return wrong_predictions, llr


# ----- Naive Bayes with tied covariance -----
def tied_naive_bayes (DTR, LTR, DTE, LTE):
    # Compute mean for each class and tied covariance matrix
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0, cov0 = utility.compute_mean(D0), utility.compute_covariance(D0)*numpy.identity(DTR.shape[0])
    mu1, cov1 = utility.compute_mean(D1), utility.compute_covariance(D1)*numpy.identity(DTR.shape[0])
    tied_cov = (cov0*D0.shape[1] + cov1*D1.shape[1]) / DTR.shape[1]

    # Compute likelihoods
    S = numpy.empty((2,DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = numpy.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu0, tied_cov))
        S[1,i] = numpy.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu1, tied_cov))
    
    # Compute log-likelihood ratio and use it to classify samples
    llr = numpy.log(S[1,:] / S[0,:])
    # predicted_labels = numpy.log(S).argmax(axis=0)
    # wrong_predictions = len(LTE) - numpy.array(predicted_labels == LTE).sum()
    predicted_labels = utility.predict_labels(llr, 0)
    wrong_predictions = utility.count_mispredictions(predicted_labels, LTE)

    return wrong_predictions, llr


# ----- Multivariate Gaussian without tied covariance -----
def mvg (DTR, LTR, DTE, LTE):
    # Compute mean and covariance for each class
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0, cov0 = utility.compute_mean(D0), utility.compute_covariance(D0)
    mu1, cov1 = utility.compute_mean(D1), utility.compute_covariance(D1)

    # Compute likelihoods
    S = numpy.empty((2,DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = numpy.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu0, cov0))
        S[1,i] = numpy.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu1, cov1))
    
    # Compute log-likelihood ratio and use it to classify samples
    llr = numpy.log(S[1,:] / S[0,:])
    # predicted_labels = numpy.log(S).argmax(axis=0)
    # wrong_predictions = len(LTE) - numpy.array(predicted_labels == LTE).sum()
    predicted_labels = utility.predict_labels(llr, 0)
    wrong_predictions = utility.count_mispredictions(predicted_labels, LTE)

    return wrong_predictions, llr


# ----- Multivariate Gaussian with tied covariance -----
def tied_mvg (DTR, LTR, DTE, LTE):
    # Compute mean for each class and tied covariance matrix
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    mu0, cov0 = utility.compute_mean(D0), utility.compute_covariance(D0)
    mu1, cov1 = utility.compute_mean(D1), utility.compute_covariance(D1)
    tied_cov = (cov0*D0.shape[1] + cov1*D1.shape[1]) / DTR.shape[1]

    # Compute likelihoods
    S = numpy.empty((2,DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = numpy.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu0, tied_cov))
        S[1,i] = numpy.exp(utility.logpdf_GAU_ND(utility.row_to_column(DTE[:,i]), mu1, tied_cov))
    
    # Compute log-likelihood ratio and use it to classify samples
    llr = numpy.log(S[1,:] / S[0,:])
    # predicted_labels = numpy.log(S).argmax(axis=0)
    # wrong_predictions = len(LTE) - numpy.array(predicted_labels == LTE).sum()
    predicted_labels = utility.predict_labels(llr, 0)
    wrong_predictions = utility.count_mispredictions(predicted_labels, LTE)

    return wrong_predictions, llr


# ----- Train model using K-Fold -----
def gen_kfold (D, L, K, pca_value, pi_value):
    classifiers = [
        (mvg, "MVG"),
        (tied_mvg, "MVG with tied covariance"),
        (naive_bayes, "Naive Bayes"),
        (tied_naive_bayes, "Naive Bayes with tied covariance")
    ]
    output_file = open(generative_training_output, "a")
    output_csv = open(output_csv_name, "a")
    N = int(D.shape[1]/K)

    if pca_value==0:
        output_file.write("No PCA, pi: %.3f\n" % (pi_value))
        print("No PCA, pi: %.3f\n" % (pi_value))
    else:
        output_file.write("PCA: %d, pi: %.3f\n" % (pca_value, pi_value))
        print("PCA: %d, pi: %.3f\n" % (pca_value, pi_value))
    
    for j,(fun,name) in enumerate(classifiers):
        wrong_predictions = 0
        numpy.random.seed(j)
        ll_ratios = []
        indexes = numpy.random.permutation(D.shape[1])

        for i in range(K):
            # Select which subset to use for evaluation
            idxTest = indexes[i*N:(i+1)*N]
            if i>0: idxTrainLeft = indexes[0:i*N]
            elif (i+1)<K: idxTrainRight = indexes[(i+1)*N:]
            if i==0: idxTrain = idxTrainRight
            elif (i+1)==K: idxTrain = idxTrainLeft
            else: idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
            DTR,LTR = D[:,idxTrain], L[idxTrain]
            DTE,LTE = D[:,idxTest], L[idxTest]

            # Apply PCA if necessary
            if pca_value!=0:
                P = dr.apply_pca(DTR, LTR, pca_value)
                DTR,DTE = numpy.dot(P.T, DTR), numpy.dot(P.T, DTE)

            # Apply classifier
            wrong, llr = fun(DTR, LTR, DTE, LTE)
            wrong_predictions += wrong
            ll_ratios.append(llr)

        # Evaluate accuracy, error rate, and minDCF
        error_rate = wrong_predictions / D.shape[1]
        accuracy = 1 - error_rate
        cost = dcf.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(ll_ratios), L)

        # Save results to file
        output_file.write("  %s\n" % (name))
        output_file.write("    Accuracy: %.3f%%\n" % (100.0*accuracy))
        output_file.write("    Error rate: %.3f%%\n" % (100.0*error_rate))
        output_file.write("    min DCF: %.3f\n" % (cost))
        output_file.write("\n")

        # Save results in CSV format
        output_csv.write("%s,%d,_,%.3f,_,_,_,_,_,%.3f,%.3f,%.5f\n" % (name, pca_value, pi_value, 100.0*accuracy, 100.0*error_rate, cost))

        # Print results to console
        print("  %s" % (name))
        print("    Accuracy: %.3f%%" % (100.0*accuracy))
        print("    Error rate: %.3f%%" % (100.0*error_rate))
        print("    min DCF: %.3f" % (cost))
        print("\n")

    output_file.close()
    output_csv.close()
    