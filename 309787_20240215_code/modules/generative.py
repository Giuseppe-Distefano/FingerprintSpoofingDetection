##### LIBRARIES #####
import numpy
import utils as utils
import math
import costs as costs


##### GLOBAL VARIABLES #####
pi_eff = 0.5
Cfn = Cfn_eff = 1
Cfp = Cfp_eff = 10
effective_prior = (pi_eff*Cfn_eff) / (pi_eff*Cfn_eff + (1-pi_eff)*Cfp_eff)
pi_values = [0.1, 0.5, 0.9, effective_prior]


##### FUNCTIONS #####
# --- Log PDF of Gaussian ---
def logpdf_GAU_ND (X, mu, C):    
    n_rows, n_cols = X.shape[0], X.shape[1]
    Y = numpy.empty([1,n_cols])
    
    for i in range(n_cols):
        x = X[:,i:i+1]
        _,det=numpy.linalg.slogdet(C)
        inv=numpy.linalg.inv(C)
        diff = x - mu
        Y[:,i] = -(n_rows/2)*numpy.log(2*math.pi) - 0.5*det - 0.5*numpy.dot(diff.T, numpy.dot(inv, diff))
    
    return Y.ravel()


# --- Log Likelihood ---
def loglikelihood (XND, mean, cov):
    return logpdf_GAU_ND(XND, mean, cov).sum()


# --- MVG ---
def mvg_classifier (DTR, LTR, DTE, LTE):
    DTR0, DTR1 = DTR[:, LTR==0], DTR[:, LTR==1]
    mu0, mu1 = utils.compute_mean(DTR0), utils.compute_mean(DTR1)
    C0, C1 = utils.compute_covariance(DTR0), utils.compute_covariance(DTR1)
   
    S = numpy.zeros((2, DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = numpy.exp(logpdf_GAU_ND(utils.vcol(DTE[:,i]), mu0, C0))
        S[1,i] = numpy.exp(logpdf_GAU_ND(utils.vcol(DTE[:,i]), mu1, C1))
    llr = numpy.log(S[1,:]/S[0,:])

    threshold = 0    
    predicted = utils.predict_labels(llr, threshold)
    wrong_predictions = utils.count_mispredictions(predicted, LTE)
    
    return wrong_predictions, llr


# --- Naive Bayes ---
def naive_bayes_classifier (DTR, LTR, DTE, LTE):
    DTR0, DTR1 = DTR[:, LTR==0], DTR[:, LTR==1]
    mu0, mu1 = utils.compute_mean(DTR0), utils.compute_mean(DTR1)
    C0= utils.compute_covariance(DTR0) * numpy.identity(DTR.shape[0])
    C1= utils.compute_covariance(DTR1) * numpy.identity(DTR.shape[0])
    
    S=numpy.empty((2, DTE.shape[1]))    
    for i in range(DTE.shape[1]):
        S[0,i] = numpy.exp(logpdf_GAU_ND(utils.vcol(DTE[:,i]), mu0, C0))
        S[1,i] = numpy.exp(logpdf_GAU_ND(utils.vcol(DTE[:,i]), mu1, C1))
    llr = numpy.log(S[1,:]/S[0,:])
    
    threshold = 0
    predicted = utils.predict_labels(llr, threshold)
    wrong_predictions = utils.count_mispredictions(predicted, LTE)
    
    return wrong_predictions, llr


# --- MVG with Tied Covariance ---
def mvg_tied_classifier (DTR, LTR, DTE, LTE):
    DTR0, DTR1 = DTR[:, LTR==0], DTR[:, LTR==1]
    mu0, mu1 = utils.compute_mean(DTR0), utils.compute_mean(DTR1)
    C0, C1 = utils.compute_covariance(DTR0), utils.compute_covariance(DTR1)
    Sw = (C0*DTR0.shape[1] + C1*DTR1.shape[1]) / DTR.shape[1]
 
    S=numpy.empty((2, DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = numpy.exp(logpdf_GAU_ND(utils.vcol(DTE[:,i]), mu0, Sw))
        S[1,i] = numpy.exp(logpdf_GAU_ND(utils.vcol(DTE[:,i]), mu1, Sw))    
    llr = numpy.log(S[1,:]/S[0,:])
    
    threshold = 0
    predicted = utils.predict_labels(llr, threshold)
    wrong_predictions = utils.count_mispredictions(predicted, LTE)
    
    return wrong_predictions, llr


# --- Naive Bayes with Tied Covariance ---
def naive_bayes_tied_classifier (DTR, LTR, DTE, LTE):
    DTR0, DTR1 = DTR[:, LTR==0], DTR[:, LTR==1]
    mu0, mu1 = utils.compute_mean(DTR0), utils.compute_mean(DTR1)
    C0= utils.compute_covariance(DTR0) * numpy.identity(DTR.shape[0])
    C1= utils.compute_covariance(DTR1) * numpy.identity(DTR.shape[0])
    Sw = (C0*DTR0.shape[1] + C1*DTR1.shape[1]) / DTR.shape[1]

    S=numpy.empty((2, DTE.shape[1]))
    for i in range(DTE.shape[1]):
        S[0,i] = numpy.exp(logpdf_GAU_ND(utils.vcol(DTE[:,i]), mu0, Sw))
        S[1,i] = numpy.exp(logpdf_GAU_ND(utils.vcol(DTE[:,i]), mu1, Sw))
    llr = numpy.log(S[1,:]/S[0,:])

    threshold = 0
    predicted = utils.predict_labels(llr, threshold)
    wrong_predictions = utils.count_mispredictions(predicted, LTE)
    
    return wrong_predictions, llr


# --- Training MVG ---
def mvg_train (D, L, K, pca_value=8):
    Dsplits, Lsplits = (numpy.array_split(D, K, axis=1)), (numpy.array_split(L, K))

    wrongs = 0
    llrs = []

    for i in range(K):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        if pca_value!=0:
            P = utils.apply_PCA(dtr, pca_value)
            dtr, dts = numpy.dot(P.T, dtr), numpy.dot(P.T, dts)
        
        wrong, llr = mvg_classifier(dtr, ltr, dts, lts)
        wrongs += wrong
        llrs.append(llr)

    for pi_value in pi_values:
        error_rate = wrongs / (D.shape[1])
        minDCF = costs.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(llrs), L)
        print(f"MVG,{pca_value},_,{pi_value},_,_,_,_,_,{error_rate},{minDCF}\n")


# --- Training Naive Bayes ---
def naive_bayes_train (D, L, K, pca_value=8):
    Dsplits, Lsplits = (numpy.array_split(D, K, axis=1)), (numpy.array_split(L, K))

    wrongs = 0
    llrs = []

    for i in range(K):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        if pca_value!=0:
            P = utils.apply_PCA(dtr, pca_value)
            dtr, dts = numpy.dot(P.T, dtr), numpy.dot(P.T, dts)
        
        wrong, llr = naive_bayes_classifier(dtr, ltr, dts, lts)
        wrongs += wrong
        llrs.append(llr)

    for pi_value in pi_values:
        error_rate = wrongs / (D.shape[1])
        minDCF = costs.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(llrs), L)
        print(f"Naive Bayes,{pca_value},_,{pi_value},_,_,_,_,_,{error_rate},{minDCF}\n")

    
# --- Training MVG with Tied Covariance ---
def mvg_tied_train (D, L, K, pca_value=8):
    Dsplits, Lsplits = (numpy.array_split(D, K, axis=1)), (numpy.array_split(L, K))

    wrongs = 0
    llrs = []

    for i in range(K):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        if pca_value!=0:
            P = utils.apply_PCA(dtr, pca_value)
            dtr, dts = numpy.dot(P.T, dtr), numpy.dot(P.T, dts)
        
        wrong, llr = mvg_tied_classifier(dtr, ltr, dts, lts)
        wrongs += wrong
        llrs.append(llr)

    for pi_value in pi_values:
        error_rate = wrongs / (D.shape[1])
        minDCF = costs.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(llrs), L)
        print(f"MVG Tied,{pca_value},_,{pi_value},_,_,_,_,_,{error_rate},{minDCF}\n")


# --- Training Naive Bayes with Tied Covariance ---
def naive_bayes_tied_train (D, L, K, pca_value=8):
    Dsplits, Lsplits = (numpy.array_split(D, K, axis=1)), (numpy.array_split(L, K))

    wrongs = 0
    llrs = []

    for i in range(K):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        if pca_value!=0:
            P = utils.apply_PCA(dtr, pca_value)
            dtr, dts = numpy.dot(P.T, dtr), numpy.dot(P.T, dts)
        
        wrong, llr = naive_bayes_tied_classifier(dtr, ltr, dts, lts)
        wrongs += wrong
        llrs.append(llr)

    for pi_value in pi_values:
        error_rate = wrongs / (D.shape[1])
        minDCF = costs.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(llrs), L)
        print(f"Naive Bayes Tied,{pca_value},_,{pi_value},_,_,_,_,_,{error_rate},{minDCF}\n")
