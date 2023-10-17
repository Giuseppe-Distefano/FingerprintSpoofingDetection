################################
#     MODULES IMPORTATIONS     #
################################
import modules.utility as utility
import numpy as np
import scipy.optimize as sopt
import modules.pca_lda as dr
import modules.costs as dcf
import numpy.linalg as npla


####################
#     FUNCTIONS    #
####################
# ----- Logistic Regression objective -----
def lr_obj_wrap (DTR, LTR, lam, pi):
    def lr_obj (v):
        w,b = v[0:-1], v[-1]
        N = DTR.shape[1]
        term1 = lam/2 * npla.norm(w)**2
        term2 = 0
        term3 = 0

        nt = DTR[:,LTR==1].shape[1]
        nf = N-nt
        pt = nt/N

        for i in range(N):
            ci = LTR[i]
            zi = 2*ci-1
            xi = DTR[:,i]
            internal_sum = b + np.dot(w.T, xi)
            internal_prod = -np.dot(zi, internal_sum)
            if LTR[i]==0: term3 += np.logaddexp(0, internal_prod)
            else: term2 += np.logaddexp(0, internal_prod)
        loss = term1 + (pi/nt)*term2 + ((1-pi)/nf)*term3
        return loss
    return lr_obj


# ----- Compute Logistic Regression scores -----
def lr_compute_scores (DTE, v):
    s = []
    w,b = v[0:-1], v[-1]
    for i in range(DTE.shape[1]):
        xt = DTE[:,i]
        s.append(b + np.dot(w.T, xt))
    return np.array(s)


# ----- Linear Logistic Regression -----
def linear_logistic_regression (DTR, LTR, DTE, LTE, lam, pi):
    x0 = np.zeros(DTR.shape[0]+1)
    lr_obj = lr_obj_wrap(DTR, LTR, lam, pi)
    (x,_,_) = sopt.fmin_l_bfgs_b(lr_obj, x0, approx_grad=True)
    
    # Compute scores
    scores = lr_compute_scores(DTE, x)
    predicted = utility.predict_labels(scores, 0)
    wrong_predictions = utility.count_mispredictions(predicted, LTE)

    return wrong_predictions, scores


# ----- Compare classifiers using K-Fold -----
def kfold (D, L, K, pca_value, pi_value, output_filename1, output_filename2):
    output_file1 = open(output_filename1, "a")
    output_file2 = open(output_filename2, "a")
    #lambda_values = [1e-6, 1e-4, 1e-3, 0.1, 1.0]
    lambda_values = [1e-6]
    Cfn = 1
    Cfp = 10
    
    output_file1.write("Linear Logistic Regression\n")
    output_file2.write("Linear Logistic Regression\n")
    for j,lam in enumerate(lambda_values):
        wrong_predictions = 0
        np.random.seed(j)
        ll_ratios = []
        indexes = np.random.permutation(D.shape[1])
        N = int(D.shape[1]/K)

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
            wrong, scores = linear_logistic_regression(DTR, LTR, DTE, LTE, lam, pi_value)
            wrong_predictions += wrong
            ll_ratios.append(scores)

        # Evaluate accuracy and error rate
        error_rate = wrong_predictions / D.shape[1]
        accuracy = 1 - error_rate
        output_file1.write("  Lambda: " + str(lam) + ", pi: " + str(pi_value) + "\n")
        output_file1.write("  Accuracy: %.3f%%\n" % (100.0*accuracy))
        output_file1.write("  Error rate: %.3f%%\n" % (100.0*error_rate))
        output_file1.write("\n")

        # Compute min DCF
        cost = dcf.compute_min_DCF(pi_value, Cfn, Cfp, np.hstack(scores), LTE)
        output_file2.write("  pi: %.3f\n" % (pi_value))
        output_file2.write("  min DCF: %.3f\n" % (cost))
        output_file2.write("\n")