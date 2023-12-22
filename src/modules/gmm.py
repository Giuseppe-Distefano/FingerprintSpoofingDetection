################################
#     MODULES IMPORTATIONS     #
################################
import modules.utility as utility
import numpy
import scipy.optimize
import modules.costs as dcf
import modules.pca_lda as dr


###########################
#     GLOBAL VARIABLES    #
###########################
gmm_training_output = "../output/Training/GMM.txt"
pi_value = 0.5
Cfn = 1
Cfp = 10


####################
#     FUNCTIONS    #
####################
# ----- Compute log-density of a GMM for a set of samples contained in matrix X -----
def logpdf_GMM (x, gmm):    
    y = []
    for weight,mu,sigma in gmm:
        lc = utility.logpdf_GAU_ND(x, mu, sigma) + numpy.log(weight)
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

        for weight,mean,sigma in gmm:
            ll = utility.logpdf_GAU_ND(D, mean, sigma) + numpy.log(weight)
            componentsLL.append(utility.column_to_row(ll))
        LL = numpy.vstack(componentsLL)

        posterior = numpy.exp(LL - scipy.special.logsumexp(LL, axis=0))
        oldLL = LL
        prevLL = scipy.special.logsumexp(LL, axis=0).sum() / D.shape[1]
        
        if oldLL is not None: deltaLL = prevLL - oldLL
        iteration += 1
        psi = 0.01
        updatedGMM = []
        for i in range(posterior.shape[0]):
            Z = posterior[i].sum()
            F = (posterior[i:i+1, :]*D).sum(1)
            S = numpy.dot((posterior[i:i+1, :])*D, D.T)
            new_weight = Z / D.shape[1]
            new_mean = utility.row_to_column(F/Z)
            new_sigma = S/Z - numpy.dot(new_mean, new_mean.T)
            
            if tied:
                c = 0
                for j in range(posterior.shape[0]):
                    Z = post[j].sum()
                    F = (post[j:j+1, :]*D).sum(1)
                    S = numpy.dot((post[j:j+1, :])*D, D.T)
                    c += (Z * (S/Z-numpy.dot(utility.row_to_column(F/Z), utility.row_to_column(F/Z).T)))
                new_sigma = 1 / D.shape[1] * c
            
            if diag: new_sigma *= numpy.eye(new_sigma.shape[0])
            
            U,s,_ = numpy.linalg.svd(new_sigma)
            s[s<psi] = psi
            new_sigma = numpy.dot(U, utility.row_to_column(s)*U.T)
            updatedGMM.append((new_weight, new_mean, new_sigma))
                
        gmm = updatedGMM
        log_LL = []
        for w,mu,C in gmm:
            ll = utility.logpdf_GAU_ND(D, mu, C) + numpy.log(w)
            log_LL.append(utility.column_to_row(ll))
        LL = numpy.vstack(log_LL)
        post = LL - scipy.special.logsumexp(LL, axis=0)
        post = numpy.exp(post)
        oldLL = prevLL
        prevLL = scipy.special.logsumexp(LL, axis=0).sum() / D.shape[1]
        deltaLL = prevLL - oldLL
            
    return gmm


# ----- ___ -----
def ML_GMM_LBG (D, weights, means, sigma, num_components, diag, tied):
    gmm = [(weights,means,sigma)]
    newGMM = []

    while len(gmm)<=num_components:
        if len(gmm)!=1: gmm = ML_GMM_iteration(D, gmm, diag, tied)
        if len(gmm)==num_components: break

        for(weight, mu, sigma) in gmm:
            U,s,_ = numpy.linalg.svd(sigma)
            s[s<0.01] = 0.01
            sigma = numpy.dot(U, utility.row_to_column(s)*U.T)
            
            newGMM.append((weight*0.5, mu+s[0]**0.5*U[:, 0:1]*0.1, sigma)) 
            newGMM.append((weight*0.5, mu-s[0]**0.5*U[:, 0:1]*0.1, sigma))

    return newGMM


# ----- GMM classifier -----
def gmm_classifier (DTR, LTR, DTE, LTE, diag, tied, l0, l1):
    # Consider only labels=0
    D0 = DTR[:,LTR==0]
    num_components_0 = len(D0)
    w0 = 1.0
    mu0 = utility.row_to_column(utility.compute_mean(D0))
    sigma0 = compute_covariance(D0)
    if diag: sigma0 *= numpy.eye(sigma0.shape[0])
    U,s,_ = numpy.linalg.svd(sigma0)
    s[s<0.01] = 0.01
    C0 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm0 = ML_GMM_LBG(D0, w0, mu0, C0, l0, diag, tied)
    _,score0 = logpdf_GMM(DTE, gmm0)

    # Consider only labels=1
    D1 = DTR[:,LTR==1]
    w1 = 1.0
    mu1 = utility.row_to_column(utility.compute_mean(D1))
    sigma1 = compute_covariance(D1)
    if diag: sigma1 *= numpy.eye(sigma1.shape[0])
    U,s,_ = numpy.linalg.svd(sigma1)
    s[s<0.01] = 0.01
    C1 = numpy.dot(U, utility.row_to_column(s)*U.T)
    gmm1 = ML_GMM_LBG(D0, w1, mu1, C1, l1, diag, tied)
    _,score1 = logpdf_GMM(DTE, gmm1)

    # Compute scores and wrong predictions
    scores = numpy.vstack((score0, score1))
    marginals = utility.column_to_row(scipy.special.logsumexp(scores, axis=0))
    f = np.exp(scores - marginals)
    wrong_predictions = (f.argmax(0)!=LTE).sum()
    scores = (score1-score0)[0]


# ----- GMM training -----
def gmm_kfold (D, L, K, pca_value, diag=False, tied=False, l0=8, l1=2):
    output_file = open(gmm_training_output, "a")
    N = int(D.shape[1]/K)

    if pca_value==0:
        output_file.write("No PCA")
        print("No PCA")
    else:
        output_file.write("PCA: %d" % (pca_value))
        print("PCA: %d" % (pca_value))

    wrong_predictions = 0
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
        wrong, score = gmm_classifier(DTR, LTR, DTE, LTE, diag, tied, l0, l1)
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

    # Print results to console
    print("  %s" % (name))
    print("    Accuracy: %.3f%%" % (100.0*accuracy))
    print("    Error rate: %.3f%%" % (100.0*error_rate))
    print("    min DCF: %.3f\n" % (cost))
    print("\n")

    output_file.close()