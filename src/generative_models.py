import numpy as np
import scipy as sc
import utility as u
import discriminative_models as lr
def logpdf_GAU_ND(x, mu, sigma):
    term1 = -(x.shape[0]/2)*np.log(2*np.pi)
    term2 = -(1/2)*(np.linalg.slogdet(sigma)[1])
    term3 = -(1/2)*((np.dot((x-mu).T, np.linalg.inv(sigma))).T*(x-mu)).sum(axis=0)
    return term1+term2+term3

def computeScoreMatrix(D, mu0, sigma0, mu1, sigma1,  callback):
    S = np.array([callback(D, mu0, sigma0), callback( D, mu1, sigma1)])
    return S

def computeLogSPost(DTE,LTE,mu0,sigma0,mu1,sigma1):
    logS = computeScoreMatrix(DTE, mu0, sigma0, mu1,sigma1, logpdf_GAU_ND)
    priorLogProbabilities = u.vcol(np.array([np.log(1/2), np.log(1/2)]))
    logSJoint = logS+priorLogProbabilities  
    logSMarginal =  u.vrow(sc.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    numberOfCorrectPredictions = 0
    predictedLabels = logSPost.argmax(axis=0)
    numberOfCorrectPredictions += np.array(predictedLabels == LTE).sum()
    return numberOfCorrectPredictions

def NaiveBayes(DTR,LTR,DTE,LTE):
    #Naive Bayes
    (mu0, sigma0), (mu1, sigma1) = u.computeMLestimates(DTR, LTR)    
    (sigma0, sigma1) = (sigma0*np.identity(sigma0.shape[0]), sigma1*np.identity(sigma1.shape[0]))
    numberOfCorrectPredictions = computeLogSPost(DTE,LTE,mu0,sigma0,mu1,sigma1)
    return numberOfCorrectPredictions


def TiedNaiveBayes(DTR,LTR,DTE,LTE):
    #Tied Covariance Naive Bayes
    (mu0, sigma0), (mu1, sigma1) = u.computeMLestimates(DTR, LTR)
    (sigma0, sigma1) = (sigma0*np.identity(sigma0.shape[0]), sigma1*np.identity(sigma1.shape[0]))
    sigma = (1/DTR.shape[1])*((LTR == 0).sum()*sigma0 +(LTR == 1).sum()*sigma1)
    numberOfCorrectPredictions = computeLogSPost(DTE,LTE,mu0,sigma,mu1,sigma)
    return numberOfCorrectPredictions

def MVG(DTR,LTR,DTE,LTE):
    (mu0, sigma0), (mu1, sigma1) = u.computeMLestimates(DTR, LTR)   
    numberOfCorrectPredictions = computeLogSPost(DTE,LTE,mu0,sigma0,mu1,sigma1)
    return numberOfCorrectPredictions


def TiedMVG(DTR,LTR,DTE,LTE):
    #Tied Covariance MVG
    (mu0, sigma0), (mu1, sigma1) = u.computeMLestimates(DTR, LTR)
    sigma = (1/DTR.shape[1])*((LTR == 0).sum()*sigma0 +(LTR == 1).sum()*sigma1)
    numberOfCorrectPredictions = computeLogSPost(DTE,LTE,mu0,sigma,mu1,sigma)
    return numberOfCorrectPredictions

def Kfold(D,L,K):
    N = int(D.shape[1]/K)
    classifiers = [(MVG, "Multivariate Gaussian Classifier"), (NaiveBayes, "Naive Bayes"), (TiedMVG, "Tied MVG Covariance"), (TiedNaiveBayes, "Tied NB Covariance"),(lr.logistic_regression,"Logistic Regression")]

    for j, (c, cstring) in enumerate(classifiers):
        nCorrectPrediction = 0
        np.random.seed(j)
        indexes = np.random.permutation(D.shape[1])
        for i in range(K):
            idxTest = indexes[i*N:(i+1)*N]
            if i > 0:
                idxTrainLeft = indexes[0:i*N]
            elif (i+1) < K:
                idxTrainRight = indexes[(i+1)*N:]

            if i == 0:
                idxTrain = idxTrainRight
            elif i == K-1:
                idxTrain = idxTrainLeft
            else:
                idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
            
            DTR = D[:, idxTrain]
            LTR = L[idxTrain]
            DTE = D[:, idxTest]
            LTE = L[idxTest]
            nCorrectPrediction += c(DTR, LTR, DTE, LTE)

        accuracy = nCorrectPrediction/D.shape[1]
        errorRate = 1 - accuracy
        print(f"{cstring} k-fold results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n") 
