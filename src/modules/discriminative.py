################################
#     MODULES IMPORTATIONS     #
################################
import modules.utility as utility
import numpy
import scipy.optimize
import modules.pca_lda as dr
import modules.costs as dcf
import numpy.linalg


####################
# GLOBAL VARIABLES #
####################
discriminative_training_output = "../output/Training/Discriminative.txt"
output_csv_name = "../output/Training/Results.csv"
Cfn = 1
Cfp = 10


####################
#     FUNCTIONS    #
####################
# ----- Logistic Regression objective -----
def lr_obj_wrap (DTR, LTR, lambda_value, pi_value):
    def lr_obj (v):
        w,b = v[0:-1], v[-1]
        N = DTR.shape[1]
        term1 = lambda_value/2 * numpy.linalg.norm(w)**2
        term2 = 0
        term3 = 0

        nt = DTR[:,LTR==1].shape[1]
        nf = N-nt

        for i in range(N):
            ci = LTR[i]
            zi = 2*ci-1
            xi = DTR[:,i]
            internal_sum = b + numpy.dot(w.T, xi)
            internal_prod = -numpy.dot(zi, internal_sum)
            if LTR[i]==0: term3 += numpy.logaddexp(0, internal_prod)
            else: term2 += numpy.logaddexp(0, internal_prod)
        loss = term1 + (pi_value/nt)*term2 + ((1-pi_value)/nf)*term3
        return loss
    return lr_obj


# ----- Compute Logistic Regression scores -----
def lr_compute_scores (DTE, v):
    s = numpy.empty((DTE.shape[1]))
    w,b = v[0:-1], v[-1]
    for i in range(DTE.shape[1]):
        xt = DTE[:,i]
        s[i] = b + numpy.dot(w.T, xt)
    return s


# ----- Numerical optimization -----
def numerical_optimization (function, x0, grad=None):
    if grad is None: x,_,_=scipy.optimize.fmin_l_bfgs_b(function,x0,fprime=numpy.gradient(function))
    else: x,_,_=scipy.optimize.fmin_l_bfgs_b(function,x0,fprime=grad)
    return x


# ----- Logistic Regression gradient -----
def lr_compute_gradient (DTR, LTR, lambda_value, pi_value):
    z = numpy.empty((LTR.shape[0]))
    z = 2*LTR-1

    def gradient (v):
        w,b = v[0:-1], v[-1]
        term1 = lambda_value * w
        term2 = 0
        term3 = 0
        
        nt = DTR[:, LTR == 1].shape[1]
        nf = DTR.shape[1]-nt
        
        for i in range(DTR.shape[1]):
            S = numpy.dot(w.T, DTR[:,i]) + b
            zi_si = numpy.dot(z[i], S)
            if LTR[i]==1: term2 += numpy.dot(numpy.exp(-zi_si),(numpy.dot(-z[i],DTR[:,i])))/(1+numpy.exp(-zi_si))
            else: term3 += numpy.dot(numpy.exp(-zi_si),(numpy.dot(-z[i],DTR[:,i])))/(1+numpy.exp(-zi_si))
        dw = term1 + (pi_value/nt)*term2 + (1-pi_value)/(nf)*term3

        term1 = 0           
        term2 = 0
        for i in range(DTR.shape[1]):
            S=numpy.dot(w.T,DTR[:,i])+b
            zi_si = numpy.dot(z[i], S)
            if LTR[i] == 1: term1 += (numpy.exp(-zi_si) * (-z[i]))/(1+numpy.exp(-zi_si))
            else: term2 += (numpy.exp(-zi_si) * (-z[i]))/(1+numpy.exp(-zi_si))
        db = (pi_value/nt)*term1 + (1-pi_value)/(nf)*term2

        return numpy.hstack((dw, db))
    return gradient


# ----- Linear Logistic Regression -----
def linear_logistic_regression (DTR, LTR, DTE, LTE, lam, pi):
    x0 = numpy.zeros(DTR.shape[0]+1)
    x = numerical_optimization(lr_obj_wrap(DTR, LTR, lam, pi), x0, lr_compute_gradient(DTR, LTR, lam, pi))
    
    # Compute scores
    scores = lr_compute_scores(DTE, x)
    predicted_labels = utility.predict_labels(scores, 0)
    wrong_predictions = utility.count_mispredictions(predicted_labels, LTE)

    return wrong_predictions, scores


# ----- Quadratic Logistic Regression -----
def quadratic_logistic_regression (DTR, LTR, DTE, LTE, lam, pi):
    DTRe = numpy.apply_along_axis(utility.square_and_transpose, 0, DTR)
    DTEe = numpy.apply_along_axis(utility.square_and_transpose, 0, DTE)
    phi_T = numpy.array(numpy.vstack([DTRe, DTR]))
    phi_E = numpy.array(numpy.vstack([DTEe, DTE]))
        
    x0 = numpy.zeros(phi_T.shape[0]+1)
    x = numerical_optimization(lr_obj_wrap(phi_T, LTR, lam, pi), x0, lr_compute_gradient(phi_T, LTR, lam, pi))

    # Compute scores and wrong predictions
    scores = lr_compute_scores(phi_E, x)
    predicted_labels = utility.predict_labels(scores, 0)
    wrong_predictions = utility.count_mispredictions(predicted_labels, LTE)

    return wrong_predictions, scores


# ----- Train model using K-Fold -----
def dis_kfold (D, L, K, pca_value, z_value, pi_value, lambda_value):
    classifiers = [
        (linear_logistic_regression, "Linear Logistic Regression"),
        (quadratic_logistic_regression, "Quadratic Logistic Regression")
    ]
    output_file = open(discriminative_training_output, "a")
    output_csv = open(output_csv_name, "a")
    N = int(D.shape[1]/K)

    if pca_value==0:
        if z_value==0:
            output_file.write("No PCA, No ZNorm, pi: %.3f, lambda: %.7f\n" % (pi_value, lambda_value))
            print("No PCA, No ZNorm, pi: %.3f, lambda: %.7f\n" % (pi_value, lambda_value))
        else:
            output_file.write("No PCA, ZNorm, pi: %.3f, lambda: %.7f\n" % (pi_value, lambda_value))
            print("No PCA, ZNorm, pi: %.3f, lambda: %.7f\n" % (pi_value, lambda_value))
    else:
        if z_value==0:
            output_file.write("PCA: %d, No ZNorm, pi: %.3f, lambda: %.7f\n" % (pca_value, pi_value, lambda_value))
            print("PCA: %d, No ZNorm, pi: %.3f, lambda: %.7f\n" % (pca_value, pi_value, lambda_value))
        else:
            output_file.write("PCA: %d, ZNorm, pi: %.3f, lambda: %.7f\n" % (pca_value, pi_value, lambda_value))
            print("PCA: %d, ZNorm, pi: %.3f, lambda: %.7f\n" % (pca_value, pi_value, lambda_value))
    
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

            # Apply ZNorm if necessary
            if z_value!=0:
                DTR,DTE = utility.compute_znorm(DTR, DTE)

            # Apply PCA if necessary
            if pca_value!=0:
                P = dr.apply_pca(DTR, LTR, pca_value)
                DTR,DTE = numpy.dot(P.T, DTR), numpy.dot(P.T, DTE)

            # Apply classifier
            wrong, scores = fun(DTR, LTR, DTE, LTE, lambda_value, pi_value)
            wrong_predictions += wrong
            ll_ratios.append(scores)

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
        output_csv.write("%s,%d,%d,%.3f,%.7f,_,_,_,_,%.3f,%.3f,%.5f\n" % (name, pca_value, z_value, pi_value, lambda_value, 100.0*accuracy, 100.0*error_rate, cost))

        # Print results to console
        print("  %s" % (name))
        print("    Accuracy: %.3f%%" % (100.0*accuracy))
        print("    Error rate: %.3f%%" % (100.0*error_rate))
        print("    min DCF: %.3f" % (cost))
        print("\n")
    
    output_file.close()
    output_csv.close()


# ----- Train a specific model -----
def train_qlr (D, L, K, pca_value, z_value, pi_value, lambda_value):
    N = int(D.shape[1]/K)

    wrong_predictions = 0
    numpy.random.seed(1)
    ll_ratios = []
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

        # Apply ZNorm if necessary
        if z_value!=0:
            DTR,DTE = utility.compute_znorm(DTR, DTE)

        # Apply PCA if necessary
        if pca_value!=0:
            P = dr.apply_pca(DTR, LTR, pca_value)
            DTR,DTE = numpy.dot(P.T, DTR), numpy.dot(P.T, DTE)

        # Apply classifier
        wrong, scores = quadratic_logistic_regression(DTR, LTR, DTE, LTE, lambda_value, pi_value)
        wrong_predictions += wrong
        ll_ratios.append(scores)
        labels.append(LTE)
    
    return numpy.hstack(ll_ratios)


# ----- Model evaluation -----
def qlr_eval (DTR, LTR, DTE, LTE, pca_value, z_value, lambda_value, pi_value):
    ll_ratios = []

    # Apply ZNorm if necessary
    if z_value!=0:
        DTR,DTE = utility.compute_znorm(DTR, DTE)
    
    # Apply PCA if necessary
    if pca_value!=0:
        P = dr.apply_pca(DTR, LTR, pca_value)
        DTR,DTE = numpy.dot(P.T, DTR), numpy.dot(P.T, DTE)
    
    _, score = quadratic_logistic_regression(DTR, LTR, DTE, LTE, lambda_value, pi_value)
    ll_ratios.append(score)

    return numpy.hstack(ll_ratios)
