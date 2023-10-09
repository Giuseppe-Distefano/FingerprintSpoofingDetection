import utility as u
import numpy as np
import scipy.optimize as sopt
import numpy.linalg as npla
##### Logistic Regression objective #####
def lr_obj_wrap (DTR, LTR, lam):
    z=np.empty((LTR.shape[0]))
    
    for i in range(LTR.shape[0]):
        z[i]=2*LTR[i]-1
    def lr_obj (v):
        w,b = v[0:-1], v[-1]
        N = DTR.shape[1]
        term1 = lam/2 * npla.norm(w)**2
        term2 = 1/N
        term3 = 0
        for i in range(N):
            ci = LTR[i]
            zi = 2*ci-1
            xi = DTR[:,i]
            term3 += np.logaddexp(0, -zi*(b + np.dot(w.T, xi)))
        loss = term1 + term2*term3
        return loss
    return lr_obj


##### Compute scores of Logistc Regression #####
def lr_compute_scores (DTE, LTE, v):
    w,b = v[0:-1], v[-1]
    n = DTE.shape[1]
    LP = []
    for i in range(0,n):
        xt = DTE[:,i]
        si = b + np.dot(w.T, xt)
        if (si>0): LP.append(1)
        else: LP.append(0)
    correctly = np.sum(np.array(LP==LTE))
    return correctly
# ----- Logistic Regression -----
def logistic_regression (DTR, LTR, DTE, LTE):
    lambda_values = [1e-6, 1e-4, 1e-2, 0.1, 1.0]
    max_correctly = 0

    for lam in lambda_values:
        lr_obj = u.lr_obj_wrap(DTR, LTR, lam)
        (x,_,_) = sopt.fmin_l_bfgs_b(lr_obj, np.zeros(DTR.shape[0]+1), approx_grad=True)
        correctly = u.lr_compute_scores(DTE, LTE, x)
        if correctly>max_correctly: max_correctly=correctly

    return max_correctly