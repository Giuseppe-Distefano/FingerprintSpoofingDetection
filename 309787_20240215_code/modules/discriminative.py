##### LIBRARIES #####
import numpy
import utils as utils
import scipy.special
import costs as costs


##### GLOBAL VARIABLES #####
pi_eff = 0.5
Cfn = Cfn_eff = 1
Cfp = Cfp_eff = 10
effective_prior = (pi_eff*Cfn_eff) / (pi_eff*Cfn_eff + (1-pi_eff)*Cfp_eff)
pi_values = [0.1, 0.5, 0.9, effective_prior]


##### FUNCTIONS #####
# --- Numerical optimization ---
def numerical_optimization (func, x0, grad=None):
    if grad is None: x,f,_ = scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=gradient(func))
    else: x,f,_ = scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=grad)
    return x,f


# --- ___ ---
def lr_obj_wrap (DTR, LTR, lambda_value):
    z = numpy.empty((LTR.shape[0]))
    for i in range(LTR.shape[0]): z[i] = 2*LTR[i]-1

    def lr_obj(v):
        w, b = v[0:-1], v[-1]
        t1 = (lambda_value/2) * (numpy.linalg.norm(w)**2)
        t2 = 0
        for i in range(DTR.shape[1]):
            s = numpy.dot(w.T,DTR[:,i])+b
            p = - numpy.dot(z[i], s)
            t2 +=numpy.logaddexp(0, p)
        t2 /= DTR.shape[1]
        
        return t1+t2

    return lr_obj


# --- ___ ---
def weighted_lr_obj_wrap (DTR, LTR, lambda_value, pi):
    z = numpy.empty((LTR.shape[0]))
    for i in range(LTR.shape[0]): z[i]=2*LTR[i]-1
    
    def lr_obj(v):
        w, b = v[0:-1], v[-1]
        t1 = (lambda_value/2) * (numpy.linalg.norm(w)**2)
      

        t2 = 0
        t3 = 0
        nt = DTR[:, LTR==1].shape[1]
        nf = DTR.shape[1] - nt


        for i in range(DTR.shape[1]):
            internal_sum=numpy.dot(w.T,DTR[:,i])+b
            
            if LTR[i] == 0:
                internal_product_0 = - numpy.dot(z[i],internal_sum)
                t3 += numpy.logaddexp(0, internal_product_0)
            else :
                internal_product_1 = - numpy.dot(z[i],internal_sum)
                t2 += numpy.logaddexp(0, internal_product_1)
        
        return t1 + (pi/nt)*t2 + (1-pi)/(nf) * t3
    
    return lr_obj


# --- ___ ---
def gradient (f):
    return numpy.gradient(f)


# --- ___ ---
def grad_test (DTR, LTR, l, pi):
    z = numpy.empty((LTR.shape[0]))
    z = 2*LTR - 1
    
    def gradient(v):
        w, b = v[0:-1], v[-1]
        
        t1 = l*w
        t2 = t3 = 0
        nt = DTR[:, LTR == 1].shape[1]
        nf = DTR.shape[1]-nt
        for i in range(DTR.shape[1]):
            S = numpy.dot(w.T,DTR[:,i]) + b
            zi_si = numpy.dot(z[i], S)

            if LTR[i] == 1: t2 += numpy.dot(numpy.exp(-zi_si), (numpy.dot(-z[i],DTR[:,i]))) / (1+numpy.exp(-zi_si))
            else : t3 += numpy.dot(numpy.exp(-zi_si), (numpy.dot(-z[i],DTR[:,i]))) / (1+numpy.exp(-zi_si))
        dw = t1 + (pi/nt)*t2 + (1-pi)/(nf) * t3
        
        t1 = t2 = 0
        for i in range(DTR.shape[1]):
            S = numpy.dot(w.T,DTR[:,i]) + b
            zi_si = numpy.dot(z[i], S)

            if LTR[i] == 1: t1 += (numpy.exp(-zi_si) * (-z[i])) / (1+numpy.exp(-zi_si))
            else : t2 += (numpy.exp(-zi_si) * (-z[i])) / (1+numpy.exp(-zi_si))
        db = (pi/nt)*t1 + (1-pi)/(nf) * t2
        
        return numpy.hstack((dw,db))

    return gradient


# --- Compute scores ---
def compute_score (DTE, v):
    s = numpy.empty((DTE.shape[1]))
    w, b = v[0:-1], v[-1]
    for i in range(DTE.shape[1]): s[i] = numpy.dot(w.T,DTE[:,i]) + b
    return s


# --- ___ ---
def compute_score_calibration (DTE, v, eff_prior):
    s = numpy.empty((DTE.shape[1]))
    w, b = v[0:-1], v[-1]
    for i in range(DTE.shape[1]): s[i] = numpy.dot(w.T,DTE[:,i]) + b - numpy.log(eff_prior/(1-eff_prior))
    return s


# --- LR calibration ---
def lr_calibration (DTR, LTR, DTE, LTE, lambda_value=0.1, pi_value=0.5):
    x0 = numpy.zeros(DTR.shape[0] + 1)
    J = weighted_lr_obj_wrap(DTR, LTR, lambda_value, pi_value)
    grad = grad_test(DTR, LTR, lambda_value, pi_value)
    v,_ = numerical_optimization(J, x0, grad)
    
    scores = compute_score_calibration(DTE, v, pi_value)
    predicted = utils.predict_labels(scores, 0)
    wrong_predictions = utils.count_mispredictions(predicted, LTE)
    
    return wrong_predictions, scores


# --- Logistic Regression classifier ---
def lr_classifier (DTR, LTR, DTE, LTE, lambda_value=0.1, pi_value=0.5):
    x0 = numpy.zeros(DTR.shape[0] + 1)
    J = weighted_lr_obj_wrap(DTR, LTR, lambda_value, pi_value)
    grad = grad_test(DTR, LTR, lambda_value, pi_value)
    v,_ = numerical_optimization(J, x0, grad)
    
    scores = compute_score(DTE,v)
    predicted = utils.predict_labels(scores, 0)
    wrong_predictions = utils.count_mispredictions(predicted, LTE)
    
    return wrong_predictions, scores


# --- Quadratic Logistic Regression classifier ---
def qlr_classifier (DTR, LTR, DTE, LTE, lambda_value=0.1, pi_value=0.5):    
    def vec_x_xT (x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
        return xxT

    dtr_e = numpy.apply_along_axis(vec_x_xT, 0, DTR)
    dte_e = numpy.apply_along_axis(vec_x_xT, 0, DTE)
    phi_r = numpy.array(numpy.vstack([dtr_e, DTR]))
    phi_e = numpy.array(numpy.vstack([dte_e, DTE]))
        
    x0 = numpy.zeros(phi_r.shape[0] + 1)
    J = weighted_lr_obj_wrap(phi_r, LTR, lambda_value, pi_value)
    grad = grad_test(phi_r, LTR, lambda_value, pi_value)
    v,_ = numerical_optimization(J, x0, grad)
    
    scores = compute_score(phi_e, v)
    predicted = utils.predict_labels(scores, 0)
    wrong_predictions = utils.count_mispredictions(predicted, LTE)
            
    return wrong_predictions, scores


# --- Training Linear Logistic Regression ---
def linear_lr_train (D, L, K, z_value, lambda_value, pi_value, pca_value=8):
    Dsplits, Lsplits = (numpy.array_split(D, K, axis=1)), (numpy.array_split(L, K))

    wrongs = 0
    llrs = []
    
    for i in range(K):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        if z_value!=0: dtr, dts = utils.compute_znorm(dtr, dts)
        if pca_value!=0:
            P = utils.apply_PCA(dtr, pca_value)
            dtr, dts = numpy.dot(P.T,dtr), numpy.dot(P.T,dts)

        wrong, llr = lr_classifier(dtr, ltr, dts, lts, lambda_value, pi_value)
        wrongs += wrong
        llrs.append(llr)

    error_rate = wrongs / (D.shape[1])
    minDCF = costs.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(llrs), L)
    
    minDCF = costs.compute_min_DCF(effective_prior, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Linear LR eff pi:,{pi_value},{minDCF}\n")
    minDCF = costs.compute_min_DCF(0.1, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Linear LR 0.1 pi:,{pi_value},{minDCF}\n")
    minDCF = costs.compute_min_DCF(0.5, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Linear LR 0.5 pi:,{pi_value},{minDCF}\n")
    minDCF = costs.compute_min_DCF(0.9, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Linear LR 0.9 pi:,{pi_value},{minDCF}\n")
    return llrs


# --- Evaluating Linear Logistic Regression ---
def linear_lr_eval (DTR, LTR, DTE, LTE, z_value, lambda_value, pi_value, pca_value=8):
    wrongs=0
    llrs = []

    if z_value!=0: DTR, DTE = utils.compute_znorm(DTR, DTE)
    if pca_value!=0:
        P = utils.apply_PCA(DTR, pca_value)
        DTR, DTE = numpy.dot(P.T,DTR), numpy.dot(P.T,DTE)

    wrong, llr = lr_classifier(DTR, LTR, DTE, LTE, lambda_value, pi_value)
    wrongs += wrong
    llrs.append(llr)

    return llrs


# --- Training Quadratic Logistic Regression ---
def quadratic_lr_train (D, L, K, z_value, lambda_value, pi_value, pca_value=8):
    Dsplits, Lsplits = (numpy.array_split(D, K, axis=1)), (numpy.array_split(L, K))

    wrongs = 0
    llrs = []
    
    for i in range(K):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        if z_value!=0: dtr, dts = utils.compute_znorm(dtr, dts)
        if pca_value!=0:
            P = utils.apply_PCA(dtr, pca_value)
            dtr, dts = numpy.dot(P.T,dtr), numpy.dot(P.T,dts)

        wrong, llr = qlr_classifier(dtr, ltr, dts, lts, lambda_value, pi_value)
        wrongs += wrong
        llrs.append(llr)

    error_rate = wrongs / (D.shape[1])
    minDCF = costs.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Quadratic LR,{pca_value},{z_value},{pi_value},{lambda_value},_,_,_,_,{error_rate},{minDCF}\n")
    

    minDCF = costs.compute_min_DCF(0.1, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Quadratic LR 0.1 pi:,{pi_value},{minDCF}\n")
    minDCF = costs.compute_min_DCF(0.5, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Quadratic LR 0.5 pi:,{pi_value},{minDCF}\n")
    minDCF = costs.compute_min_DCF(0.9, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Quadratic LR 0.9 pi:,{pi_value},{minDCF}\n")
    return llrs


# --- Calibrating Linear Logistic Regression ---
def linear_lr_train_calibration (D, L, K, lambda_value, pi_value):
    Dsplits, Lsplits = (numpy.array_split(D, K, axis=1)), (numpy.array_split(L, K))

    wrongs = 0
    llrs = []
    
    for i in range(K):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        wrong, llr = lr_calibration(dtr, ltr, dts, lts, lambda_value, pi_value)
        wrongs += wrong
        llrs.append(llr)

    return llrs


# --- Evaluating Quadratic Logistic Regression ---
def quadratic_lr_evaluation (DTR, LTR, DTE, LTE, z_value,lambda_value, pi_value, pca_value=8):
    llrs = []
    
    if z_value!=0: DTR, DTE = utils.compute_znorm(DTR, DTE)
    if pca_value!=0:
        P= utils.apply_PCA(DTR, pca_value)
        DTR, DTR = numpy.dot(P.T, DTR), numpy.dot(P.T, DTE)
        
    _,llr = qlr_classifier(DTR, LTR, DTE, LTE, lambda_value, pi_value)
    llrs.append(llr)
    
    return llrs
