##### MODULES IMPORTATIONS #####
import modules.utility as utility
import numpy as np
import scipy.optimize as sopt


# ----- Naive Bayes without tied covariance -----
def naive_bayes (DTR, LTR, DTE, LTE):
    mu, sigma = utility.compute_muc_sigmac(DTR, LTR)

    sigma[0] = sigma[0] * np.identity(sigma[0].shape[0])
    sigma[1] = sigma[1] * np.identity(sigma[1].shape[0])
    S = utility.compute_likelihoods(DTE, LTE, mu, sigma)
    Pc = 1.0/2.0
    logSPost = utility.compute_logposterior(S, Pc)

    correctly = utility.compute_correct_predictions(logSPost, LTE)
    return correctly


# ----- Naive Bayes with tied covariance -----
def tied_naive_bayes (DTR, LTR, DTE, LTE):
    mu, sigma = utility.compute_muc_sigma(DTR, LTR)
    
    (sigma[0], sigma[1]) = (sigma[0]*np.identity(sigma[0].shape[0]), sigma[1]*np.identity(sigma[1].shape[0]))
    sigma = (1/DTR.shape[1])*((LTR == 0).sum()*sigma[0] +(LTR == 1).sum()*sigma[1])
    S = utility.compute_tied_likelihoods(DTE, LTE, mu, sigma)
    Pc = 1.0/2.0
    logSPost = utility.compute_logposterior(S, Pc)
    
    correctly = utility.compute_correct_predictions(logSPost, LTE)
    return correctly


# ----- Multivariate Gaussian without tied covariance -----
def mvg (DTR, LTR, DTE, LTE):
    mu, sigma = utility.compute_muc_sigmac(DTR, LTR)
    
    S = utility.compute_likelihoods(DTE, LTE, mu, sigma)
    Pc = 1.0/2.0
    logSPost = utility.compute_logposterior(S, Pc)
    
    correctly = utility.compute_correct_predictions(logSPost, LTE)
    return correctly


# ----- Multivariate Gaussian with tied covariance -----
def tied_mvg (DTR, LTR, DTE, LTE):
    mu, sigma = utility.compute_muc_sigma(DTR, LTR)
    
    #(sigma[0], sigma[1]) = (sigma[0]*np.identity(sigma[0].shape[0]), sigma[1]*np.identity(sigma[1].shape[0]))
    sigma = (1/DTR.shape[1])*((LTR == 0).sum()*sigma[0] +(LTR == 1).sum()*sigma[1])
    S = utility.compute_tied_likelihoods(DTE, LTE, mu, sigma)
    Pc = 1.0/2.0
    logSPost = utility.compute_logposterior(S, Pc)
    
    correctly = utility.compute_correct_predictions(logSPost, LTE)
    return correctly


# ----- Logistic Regression -----
def logistic_regression (DTR, LTR, DTE, LTE):
    lambda_values = [1e-6]
    max_correctly = 0

    for lam in lambda_values:
        lr_obj = utility.lr_obj_wrap(DTR, LTR, lam)
        (x,_,_) = sopt.fmin_l_bfgs_b(lr_obj, np.zeros(DTR.shape[0]+1), approx_grad=True)
        correctly = utility.lr_compute_scores(DTE, LTE, x)
        if correctly>max_correctly: max_correctly=correctly

    return max_correctly


##### Compare classifiers using K-Fold #####
def kfold (D, L, K):
    N = int(D.shape[1]/K)
    classifiers = [
        (mvg, "Multivariate Gaussian classifier without tied covariance"),
        (naive_bayes, "Naive Bayes without tied covariance"),
        (tied_mvg, "Multivariate Gaussian classifier with tied covariance"),
        (tied_naive_bayes, "Naive Bayes with tied covariance"),
        (logistic_regression, "Logistic Regression")
    ]
    output_file = "../output/04_Kfold/evaluations.txt"
    file = open(output_file, "w+")

    for j, (fun, name) in enumerate(classifiers):
        wrong_predictions = 0
        np.random.seed(j)
        indexes = np.random.permutation(D.shape[1])
        
        for i in range(K):
            idxTest = indexes[i*N:(i+1)*N]
            if i > 0: idxTrainLeft = indexes[0:i*N]
            elif (i+1) < K: idxTrainRight = indexes[(i+1)*N:]

            if i == 0: idxTrain = idxTrainRight
            elif i == K-1: idxTrain = idxTrainLeft
            else: idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
            
            DTR = D[:, idxTrain]
            LTR = L[idxTrain]
            DV = D[:, idxTest]
            LV = L[idxTest]
            correct_predictions = fun(DTR, LTR, DV, LV)
            wrong_predictions += DV.shape[1]-correct_predictions

        error_rate = wrong_predictions/D.shape[1]
        accuracy = 1 - error_rate
        file.write("%s\n" % name)
        file.write("  Accuracy: %.3f %%\n" % (100.0*accuracy))
        file.write("  Error rate: %.3f %%\n" % (100.0*error_rate))
        file.write("\n")
    
    file.close()
