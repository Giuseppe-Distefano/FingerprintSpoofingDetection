################################
#     MODULES IMPORTATIONS     #
################################
import modules.utility as utility
import numpy
import scipy.optimize
import modules.costs as dcf
import modules.pca_lda as dr
import math


###########################
#     GLOBAL VARIABLES    #
###########################
gmm_training_output = "../output/Training/GMM.txt"
output_csv_name = "../output/Training/Results.csv"
pi_value = 0.5
Cfn = 1
Cfp = 10


####################
#     FUNCTIONS    #
####################
# ----- ___ -----
def logpdf_GAU_ND_GMM(X, mu, C):
    M,N = X.shape[0], X.shape[1]
    Y = numpy.empty([1,N])
    
    for i in range(N):
        x = X[:,i:i+1]
        _,det = numpy.linalg.slogdet(C)
        inv = numpy.linalg.inv(C)
        density = -(M/2)*numpy.log(2*math.pi) - 1/2*det - 1/2*numpy.dot((x-mu).T, numpy.dot(inv, (x-mu)))
        Y[:,i]=density
        
    return Y.ravel()


# ----- Compute log-density of a GMM for a set of samples contained in matrix X -----
def logpdf_GMM (x, gmm):    
    y = []
    for weight,mu,sigma in gmm:
        lc = logpdf_GAU_ND_GMM(x, mu, sigma) + numpy.log(weight)
        y.append(utility.column_to_row(lc))
    S = numpy.vstack(y)
    logdensity = scipy.special.logsumexp(y, axis=0)
    return S,logdensity


# ----- ___ -----
def ML_GMM_iteration(D, gmm, diag, tied):
    prevLL = None
    oldLL = None
    deltaLL = 1.0
    iteration = 0

    while deltaLL>1e-6:
        componentsLL = []
        for w, mu, C in gmm:
            ll = logpdf_GAU_ND_GMM(D, mu, C) + numpy.log(w)
            componentsLL.append(utility.column_to_row(ll))
        LL = numpy.vstack(componentsLL)
        post = numpy.exp(LL - scipy.special.logsumexp(LL, axis=0))
        oldLL = prevLL
        prevLL = scipy.special.logsumexp(LL, axis=0).sum() / D.shape[1]
        
        if oldLL is not None: deltaLL = prevLL - oldLL
        iteration += 1
        psi = 0.01
        updatedGMM = []
        for i in range(post.shape[0]):
            Z = post[i].sum()
            F = (post[i:i+1, :]*D).sum(1)
            S = numpy.dot((post[i:i+1, :])*D, D.T)
            new_weight = Z / D.shape[1]
            new_mean = utility.row_to_column(F/Z)
            new_sigma = S/Z - numpy.dot(new_mean, new_mean.T)
            
            if tied:
                c = 0
                for j in range(post.shape[0]):
                    Z = post[j].sum()
                    F = (post[j:j+1, :]*D).sum(1)
                    S = numpy.dot((post[j:j+1, :])*D, D.T)
                    c += Z * (S/Z - numpy.dot(utility.row_to_column(F/Z), utility.row_to_column(F/Z).T))
                new_sigma = c / D.shape[1]
            
            if diag: new_sigma = new_sigma * numpy.eye(new_sigma.shape[0])
            
            U, s, _ = numpy.linalg.svd(new_sigma)
            s[s<psi] = psi
            new_sigma=numpy.dot(U, utility.row_to_column(s)*U.T)
            updatedGMM.append((new_weight, new_mean, new_sigma))

        gmm = updatedGMM
        componentsLL = []
        for w, mu, C in gmm:
            ll = logpdf_GAU_ND_GMM(D, mu, C) + numpy.log(w)
            componentsLL.append(utility.column_to_row(ll))
        LL = numpy.vstack(componentsLL)
        post = LL - scipy.special.logsumexp(LL, axis=0)
        post = numpy.exp(post)
        oldLL = prevLL
        prevLL = scipy.special.logsumexp(LL, axis=0).sum() / D.shape[1]
        deltaLL = prevLL - oldLL
    
    return gmm


# ----- ___ -----
def ML_GMM_LBG (D, weights, means, sigma, G, diag, tied):
    gmm = [(weights,means,sigma)]

    while len(gmm)<=G:
        if len(gmm)!=1: gmm = ML_GMM_iteration(D, gmm, diag, tied)
        if len(gmm)==G: break

        newGMM = []
        for(weight, mu, sigma) in gmm:
            U,s,_ = numpy.linalg.svd(sigma)
            s[s<0.01] = 0.01
            sigma = numpy.dot(U, utility.row_to_column(s)*U.T)
            
            newGMM.append((weight*0.5, mu+s[0]**0.5*U[:, 0:1]*0.1, sigma)) 
            newGMM.append((weight*0.5, mu-s[0]**0.5*U[:, 0:1]*0.1, sigma))
        gmm = newGMM

    return gmm


# ----- GMM classifier - Full covariance -----
def gmm_full_covariance (DTR, LTR, DTE, LTE, g0, g1):
    # Consider only labels=0
    D0 = DTR[:,LTR==0]
    w0 = 1.0
    mu0 = utility.row_to_column(utility.compute_mean(D0))
    sigma0 = utility.compute_covariance(D0)
    U,s,_ = numpy.linalg.svd(sigma0)
    s[s<0.01] = 0.01
    C0 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm0 = ML_GMM_LBG(D0, w0, mu0, C0, g0, False, False)
    _,score0 = logpdf_GMM(DTE, gmm0)

    # Consider only labels=1
    D1 = DTR[:,LTR==1]
    w1 = 1.0
    mu1 = utility.row_to_column(utility.compute_mean(D1))
    sigma1 = utility.compute_covariance(D1)
    U,s,_ = numpy.linalg.svd(sigma1)
    s[s<0.01] = 0.01
    C1 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm1 = ML_GMM_LBG(D1, w1, mu1, C1, g1, False, False)
    _,score1 = logpdf_GMM(DTE, gmm1)

    # Compute scores and wrong predictions
    scores = numpy.vstack((score0, score1))
    marginals = utility.column_to_row(scipy.special.logsumexp(scores, axis=0))
    f = numpy.exp(scores - marginals)
    wrong_predictions = (f.argmax(0)!=LTE).sum()
    scores = (score1-score0)[0]

    return wrong_predictions, scores


# ----- GMM classifier - Diagonal covariance -----
def gmm_diagonal_covariance (DTR, LTR, DTE, LTE, g0, g1):
    # Consider only labels=0
    D0 = DTR[:,LTR==0]
    w0 = 1.0
    mu0 = utility.row_to_column(utility.compute_mean(D0))
    sigma0 = utility.compute_covariance(D0)
    sigma0 *= numpy.eye(sigma0.shape[0])
    U,s,_ = numpy.linalg.svd(sigma0)
    s[s<0.01] = 0.01
    C0 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm0 = ML_GMM_LBG(D0, w0, mu0, C0, g0, True, False)
    _,score0 = logpdf_GMM(DTE, gmm0)

    # Consider only labels=1
    D1 = DTR[:,LTR==1]
    w1 = 1.0
    mu1 = utility.row_to_column(utility.compute_mean(D1))
    sigma1 = utility.compute_covariance(D1)
    sigma1 *= numpy.eye(sigma1.shape[0])
    U,s,_ = numpy.linalg.svd(sigma1)
    s[s<0.01] = 0.01
    C1 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm1 = ML_GMM_LBG(D1, w1, mu1, C1, g1, True, False)
    _,score1 = logpdf_GMM(DTE, gmm1)

    # Compute scores and wrong predictions
    scores = numpy.vstack((score0, score1))
    marginals = utility.column_to_row(scipy.special.logsumexp(scores, axis=0))
    f = numpy.exp(scores - marginals)
    wrong_predictions = (f.argmax(0)!=LTE).sum()
    scores = (score1-score0)[0]

    return wrong_predictions, scores


# ----- GMM classifier - Tied covariance -----
def gmm_tied_covariance (DTR, LTR, DTE, LTE, g0, g1):
    # Consider only labels=0
    D0 = DTR[:,LTR==0]
    w0 = 1.0
    mu0 = utility.row_to_column(utility.compute_mean(D0))
    sigma0 = utility.compute_covariance(D0)
    U,s,_ = numpy.linalg.svd(sigma0)
    s[s<0.01] = 0.01
    C0 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm0 = ML_GMM_LBG(D0, w0, mu0, C0, g0, False, True)
    _,score0 = logpdf_GMM(DTE, gmm0)

    # Consider only labels=1
    D1 = DTR[:,LTR==1]
    w1 = 1.0
    mu1 = utility.row_to_column(utility.compute_mean(D1))
    sigma1 = utility.compute_covariance(D1)
    U,s,_ = numpy.linalg.svd(sigma1)
    s[s<0.01] = 0.01
    C1 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm1 = ML_GMM_LBG(D1, w1, mu1, C1, g1, False, True)
    _,score1 = logpdf_GMM(DTE, gmm1)

    # Compute scores and wrong predictions
    scores = numpy.vstack((score0, score1))
    marginals = utility.column_to_row(scipy.special.logsumexp(scores, axis=0))
    f = numpy.exp(scores - marginals)
    wrong_predictions = (f.argmax(0)!=LTE).sum()
    scores = (score1-score0)[0]

    return wrong_predictions, scores


# ----- GMM classifier - Tied diagonal covariance -----
def gmm_tied_diagonal_covariance (DTR, LTR, DTE, LTE, g0, g1):
    # Consider only labels=0
    D0 = DTR[:,LTR==0]
    w0 = 1.0
    mu0 = utility.row_to_column(utility.compute_mean(D0))
    sigma0 = utility.compute_covariance(D0)
    sigma0 *= numpy.eye(sigma0.shape[0])
    U,s,_ = numpy.linalg.svd(sigma0)
    s[s<0.01] = 0.01
    C0 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm0 = ML_GMM_LBG(D0, w0, mu0, C0, g0, True, True)
    _,score0 = logpdf_GMM(DTE, gmm0)

    # Consider only labels=1
    D1 = DTR[:,LTR==1]
    w1 = 1.0
    mu1 = utility.row_to_column(utility.compute_mean(D1))
    sigma1 = utility.compute_covariance(D1)
    sigma1 *= numpy.eye(sigma1.shape[0])
    U,s,_ = numpy.linalg.svd(sigma1)
    s[s<0.01] = 0.01
    C1 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm1 = ML_GMM_LBG(D1, w1, mu1, C1, g1, True, True)
    _,score1 = logpdf_GMM(DTE, gmm1)

    # Compute scores and wrong predictions
    joint = numpy.vstack((score0, score1))
    marginals = utility.column_to_row(scipy.special.logsumexp(joint, axis=0))
    f = numpy.exp(joint - marginals)
    wrong_predictions = (f.argmax(0)!=LTE).sum()
    scores = (score1-score0)[0]

    return wrong_predictions, scores


# ----- GMM training -----
def gmm_kfold (D, L, K, pca_value, g0_value, g1_value):
    classifiers = [
        (gmm_full_covariance, "GMM Full Covariance"),
        (gmm_diagonal_covariance, "GMM Diagonal Covariance"),
        (gmm_tied_covariance, "GMM Tied Covariance"),
        (gmm_tied_diagonal_covariance, "GMM Tied Diagonal Covariance")
    ]
    output_file = open(gmm_training_output, "a")
    output_csv = open(output_csv_name, "a")
    N = int(D.shape[1]/K)

    if pca_value==0:
        output_file.write("No PCA, G0: %d, G1: %d\n" % (g0_value, g1_value))
        print("No PCA, G0: %d, G1: %d\n" % (g0_value, g1_value))
    else:
        output_file.write("PCA: %d, G0: %d, G1: %d\n" % (pca_value, g0_value, g1_value))
        print("PCA: %d, G0: %d, G1: %d\n" % (pca_value, g0_value, g1_value))

    for j,(fun,name) in enumerate(classifiers):
        wrong_predictions = 0
        numpy.random.seed(j)
        scores = []
        labels = []
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
            wrong, score = fun(DTR, LTR, DTE, LTE, g0_value, g1_value)
            wrong_predictions += wrong
            scores.append(score)
            labels.append(LTE)
        
        # Evaluate accuracy and error rate
        error_rate = wrong_predictions / D.shape[1]
        accuracy = 1 - error_rate
        cost = dcf.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(scores), numpy.hstack(labels))

        # Save results to file
        output_file.write("  %s\n" % (name))
        output_file.write("    Accuracy: %.3f%%\n" % (100.0*accuracy))
        output_file.write("    Error rate: %.3f%%\n" % (100.0*error_rate))
        output_file.write("    min DCF: %.3f\n" % (cost))
        output_file.write("\n")

        # Save results in CSV format
        output_csv.write("%s,%d,_,_,_,_,%d,%d,%.3f,%.3f,%.5f\n" % (name, pca_value, g0_value, g1_value, 100.0*accuracy, 100.0*error_rate, cost))

        # Print results to console
        print("  %s" % (name))
        print("    Accuracy: %.3f%%" % (100.0*accuracy))
        print("    Error rate: %.3f%%" % (100.0*error_rate))
        print("    min DCF: %.3f\n" % (cost))
        print("\n")

    output_file.close()
    output_csv.close()


# ----- Train a specific model -----
def train_diagonal_gmm (D, L, K, pca_value, g0_value, g1_value):
    N = int(D.shape[1]/K)

    wrong_predictions = 0
    numpy.random.seed(1)
    scores = []
    labels = []
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
        wrong, score = gmm_diagonal_covariance(DTR, LTR, DTE, LTE, g0_value, g1_value)
        wrong_predictions += wrong
        scores.append(score)
        labels.append(LTE)

    return numpy.hstack(scores)
    