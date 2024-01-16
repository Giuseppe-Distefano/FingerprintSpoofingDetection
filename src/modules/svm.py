################################
#     MODULES IMPORTATIONS     #
################################
import modules.utility as utility
import numpy
import scipy.optimize
import modules.costs as dcf
import modules.pca_lda as dr


####################
# GLOBAL VARIABLES #
####################
svm_training_output = "../output/Training/SVM.txt"
output_csv_name = "../output/Training/Results.csv"
pi_value = 0.5
Cfn = 1
Cfp = 10


####################
#     FUNCTIONS    #
####################
# ----- ___ -----
def compute_modified_H (D, L, K):
    # Distance
    row = numpy.zeros(D.shape[1]) + K
    d = numpy.vstack([D, row])
    
    # Assign a class label to each sample
    z = numpy.zeros(len(L))
    for i in range(len(L)):
        if (L[i]==0): z[i] = -1
        else: z[i] = 1
    
    G_ij = numpy.dot(d.T, d)
    z_ij = numpy.dot(utility.row_to_column(z), utility.column_to_row(z))
    H = z_ij * G_ij

    return d,z,H


# ----- ___ -----
def compute_kernel_H (D, L, K):
    # Assign a class label to each sample
    z = numpy.zeros(len(L))
    for i in range(len(L)):
        if (L[i]==0): z[i] = -1
        else: z[i] = 1
    
    ker = (numpy.dot(D.T, D)+1)**2 + K**2
    p1 = numpy.dot(utility.row_to_column(z), utility.column_to_row(z))
    H = p1 * ker

    return z,H


# ----- ___ -----
def compute_polynomial_kernel_H (D, L, c, d, K):
    # Assign a class label to each sample
    z = numpy.zeros(len(L))
    for i in range(len(L)):
        if (L[i]==0): z[i] = -1
        else: z[i] = 1
    
    ker = (numpy.dot(D.T, D)+c)**d + K**2
    p1 = numpy.dot(utility.row_to_column(z), utility.column_to_row(z))
    H = p1 * ker

    return z,H


# ----- ___ -----
def compute_radial_kernel_H (D, L, gamma, K):
    # Assign a class label to each sample
    z = numpy.zeros(len(L))
    for i in range(len(L)):
        if (L[i]==0): z[i] = -1
        else: z[i] = 1

    ker = numpy.zeros((D.shape[1], D.shape[1]))
    for i in range(D.shape[1]):
        for j in range(D.shape[1]):
            ker[i,j] = numpy.exp(-gamma * (numpy.linalg.norm(D[:,i]-D[:,j])**2)) + K**2
    p1 = numpy.dot(utility.row_to_column(z), utility.column_to_row(z))
    H = p1 * ker

    return z,H


# ----- Compute gradient -----
def compute_gradient (a, H):
    p1 = numpy.dot(H, utility.row_to_column(a))
    p2 = numpy.dot(utility.column_to_row(a), p1)
    s = a.sum()
    return 0.5*p2.ravel()-s, p1.ravel()-numpy.ones(a.size)


# ----- Optimal dual solution -----
def minimize_dual (D, z, H, C):
    bounds = [(0,C)] * D.shape[1]
    alpha, dual, _ = scipy.optimize.fmin_l_bfgs_b(compute_gradient, numpy.zeros(D.shape[1]), args=(H,), bounds=bounds, factr=1.0)
    return alpha, -dual


# ----- Find optimal primal solution -----
def primal_model (D, alpha, z, K, DTE, LTE):
    w = numpy.dot(D, utility.row_to_column(alpha) * utility.row_to_column(z))
    S = numpy.dot(utility.column_to_row(w), D)
    loss = numpy.maximum(numpy.zeros(S.shape), 1-z*S).sum()

    row = numpy.zeros(DTE.shape[1]) + K
    DTEe = numpy.vstack([DTE, row])
    Se = numpy.dot(w.T, DTEe)
    predicted_labels = 1 * (Se>0)
    wrong = numpy.array(predicted_labels!=LTE).sum()

    return wrong, Se.ravel()


# ----- ___ -----
def primal_model_kernel (DTR, alpha, z, K, DTE, LTE):
    S = numpy.sum(numpy.dot(utility.column_to_row(alpha*z), (numpy.dot(DTR.T, DTE)+1)**2 + K), axis=0)
    predicted_labels = 1 * (S>0)
    wrong = numpy.array(predicted_labels!=LTE).sum()

    return wrong, S.ravel()


# ----- Linear SVM ---
def linear_svm (DTR, LTR, DTE, LTE, K, C, gamma):
    D,z,H = compute_modified_H(DTR, LTR, K)
    alpha, dual = minimize_dual(D, z, H, C)
    wrong, scores = primal_model(D, alpha, z, K, DTE, LTE)
    return wrong, scores


# ----- Quadratic SVM -----
def quadratic_svm (DTR, LTR, DTE, LTE, K, C, gamma):
    z,H = compute_kernel_H(DTR, LTR, K)
    alpha, dual = minimize_dual(DTR, z, H, C)
    wrong, scores = primal_model_kernel(DTR, alpha, z, K, DTE, LTE)
    return wrong, scores


# ----- Polynomial Kernel SVM -----
def polynomial_kernel_svm (DTR, LTR, DTE, LTE, K, C, gamma):
    c = 1
    d = 2
    z,H = compute_polynomial_kernel_H(DTR, LTR, c, d, K)
    alpha, dual = minimize_dual(DTR, z, H, C)
    scores = numpy.sum(numpy.dot(utility.column_to_row(alpha*z), (numpy.dot(DTR.T, DTE)+c)**d+K), axis=0)
    predicted_labels = 1*(scores>0)
    wrong = numpy.array(predicted_labels!=LTE).sum()
    return wrong, scores


# ----- Radial Basis Function Kernel SVM -----
def radial_kernel_svm (DTR, LTR, DTE, LTE, K, C, gamma):
    z,H = compute_radial_kernel_H(DTR, LTR, gamma, K)
    alpha, dual = minimize_dual(DTR, z, H, C)
    ker = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            ker[i,j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:,i]-DTE[:,j])**2)) + K**2
    scores = numpy.sum(numpy.dot((utility.column_to_row(alpha*z)), ker), axis=0)
    predicted_labels = 1*(scores>0)
    wrong = numpy.array(predicted_labels!=LTE).sum()
    return wrong, scores


# ----- Train model using K-Fold -----
def svm_kfold (D, L, K, pca_value, C_value, gamma_value):
    classifiers = [
        (linear_svm, "Linear SVM"),
        (quadratic_svm, "Quadratic SVM"),
        (polynomial_kernel_svm, "Polynomial Kernel SVM"),
        (radial_kernel_svm, "Radial Basis Function Kernel SVM")
    ]
    output_file = open(svm_training_output, "a")
    output_csv = open(output_csv_name, "a")
    N = int(D.shape[1]/K)

    if pca_value==0:
        output_file.write("No PCA, C: %.5f, gamma: %.5f\n" % (C_value, gamma_value))
        print("No PCA, C: %.5f, gamma: %.5f\n" % (C_value, gamma_value))
    else:
        output_file.write("PCA: %d, C: %.5f, gamma: %.5f\n" % (pca_value, C_value, gamma_value))
        print("PCA: %d, C: %.5f, gamma: %.5f\n" % (pca_value, C_value, gamma_value))

    for j,(fun,name) in enumerate(classifiers):
        wrong_predictions = 0
        numpy.random.seed(j)
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

            # Apply PCA if necessary
            if pca_value!=0:
                P = dr.apply_pca(DTR, LTR, pca_value)
                DTR,DTE = numpy.dot(P.T, DTR), numpy.dot(P.T, DTE)
            
            # Apply classifier
            wrong, scores = fun(DTR, LTR, DTE, LTE, K, C_value, gamma_value)
            wrong_predictions += wrong
            ll_ratios.append(scores)
            labels.append(LTE)
        
        # Evaluate accuracy, error rate, and minDCF
        error_rate = wrong_predictions / D.shape[1]
        accuracy = 1 - error_rate
        cost = dcf.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(ll_ratios), numpy.hstack(labels))

        # Save results to file
        output_file.write("  %s\n" % (name))
        output_file.write("    Accuracy: %.3f%%\n" % (100.0*accuracy))
        output_file.write("    Error rate: %.3f%%\n" % (100.0*error_rate))
        output_file.write("    min DCF: %.3f\n" % (cost))
        output_file.write("\n")

        # Save results in CSV format
        output_csv.write("%s,%d,_,_,%.5f,%.5f,_,_,%.3f,%.3f,%.5f\n" % (name, pca_value, C_value, gamma_value, 100.0*accuracy, 100.0*error_rate, cost))

        # Print results to console
        print("  %s" % (name))
        print("    Accuracy: %.3f%%" % (100.0*accuracy))
        print("    Error rate: %.3f%%" % (100.0*error_rate))
        print("    min DCF: %.3f" % (cost))
        print("\n")

    output_file.close()
    output_csv.close()


# ----- Train a specific model -----
def train_pol_svm (D, L, K, pca_value, C_value, gamma_value):
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

        # Apply PCA if necessary
        if pca_value!=0:
            P = dr.apply_pca(DTR, LTR, pca_value)
            DTR,DTE = numpy.dot(P.T, DTR), numpy.dot(P.T, DTE)
            
        # Apply classifier
        wrong, scores = polynomial_kernel_svm(DTR, LTR, DTE, LTE, K, C_value, gamma_value)
        wrong_predictions += wrong
        ll_ratios.append(scores)
        labels.append(LTE)

    return numpy.hstack(ll_ratios)
    