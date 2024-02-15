##### LIBRARIES #####
import numpy
import scipy
import utils 
import costs 


##### GLOBAL VARIABLES #####
pi_eff = 0.5
Cfn = Cfn_eff = 1
Cfp = Cfp_eff = 10
effective_prior = (pi_eff*Cfn_eff) / (pi_eff*Cfn_eff + (1-pi_eff)*Cfp_eff)
pi_values = [0.1, 0.5, 0.9, effective_prior]


##### FUNCTIONS #####
# --- ___ ---
def compute_mod_H (DTR, LTR, K):
    kr = numpy.zeros(DTR.shape[1]) + K
    D = numpy.vstack([DTR, kr])    
    G_ij = numpy.dot(D.T, D)
    
    z=numpy.zeros(len(LTR))
    for i in range(len(LTR)):
        if(LTR[i]==0): z[i] = -1
        else: z[i] = 1
    z_ij = numpy.dot(utils.vcol(z), utils.vrow(z))
    H = z_ij * G_ij
    
    return D, z, H


# --- Compute scores and error for kernel polynomial SVM ---
def scores_error_kernel_poly (alpha_opt, z, DTR, DTE, LTE, c, d, K):
    S = numpy.sum(numpy.dot((alpha_opt*z).reshape(1, DTR.shape[1]), (numpy.dot(DTR.T, DTE)+c)**d+ K), axis=0)
    LP = 1*(S > 0)
    wrong_predictions = LTE.size - numpy.array(LP==LTE).sum()
    return S, wrong_predictions


# --- Compute scores and error for radial basis function SVM ---
def scores_error_kernel_RBF (alpha_opt, z, DTR, DTE, LTE, K, gamma):
    kf = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kf[i,j] = RBF(DTR[:, i], DTE[:, j], gamma, K)
            
    S = numpy.sum(numpy.dot((alpha_opt*z).reshape(1, DTR.shape[1]), kf), axis=0)
    LP = 1*(S > 0)
    wrong_predictions = LTE.size - numpy.array(LP==LTE).sum()
    
    return S, wrong_predictions


# --- Gradient for the dual problem ---
def compute_dual_grad (alpha, H):
    Ha = numpy.dot(H, utils.vcol(alpha))
    aHa = numpy.dot(utils.vrow(alpha), Ha)
    a1 = alpha.sum()
    return 0.5*aHa.ravel()-a1, Ha.ravel()-numpy.ones(alpha.size)


# --- Module minimization for the dual problem ---
def minimize_mod_dual (D, C, z, H):
    bounds = [(0,C)]*D.shape[1]
    alpha_opt, opt_dual_value,_=scipy.optimize.fmin_l_bfgs_b(compute_dual_grad, numpy.zeros(D.shape[1]), args=(H,), bounds=bounds, factr=1.0)
    return alpha_opt, -opt_dual_value


# --- ___ ---
def minimize_mod_dual_balanced (D, L, C, z, H, pi_value):
    emp_pi_t, emp_pi_f = 0.34408602150537637, 0.6559139784946236
    
    bounds=[]
    for i in range(D.shape[1]):
        if(L[i]==0): bounds.append((0,C*emp_pi_f/(1-pi_value)))
        else: bounds.append((0,C*emp_pi_t/(pi_value)))

    alpha_opt,opt_dual_value,_=scipy.optimize.fmin_l_bfgs_b(compute_dual_grad, numpy.zeros(D.shape[1]), args=(H,), bounds=bounds, factr=1.0)
    return alpha_opt, -opt_dual_value


# --- SVM primal model ---
def primal_model (D, alpha_opt, z, K, C, DTE, LTE):
    w_opt = numpy.dot(D, utils.vcol(alpha_opt) * utils.vcol(z))
    S = numpy.dot(utils.vrow(w_opt), D)
    loss = numpy.maximum(numpy.zeros(S.shape), 1-z*S).sum()

    row = numpy.zeros(DTE.shape[1]) + K
    Ste = numpy.dot(w_opt.T, numpy.vstack([DTE, row]))
    LP = 1*(Ste > 0)
    
    wrong_predictions = LTE.size - numpy.array(LP==LTE).sum()
    
    return 0.5*numpy.linalg.norm(w_opt)**2 + C*loss, wrong_predictions, Ste.ravel()


# --- ___ ---
def kernel_poly_H (DTR, LTR, c, d, K):    
    z=numpy.zeros(len(LTR))
    for i in range(len(LTR)):
        if(LTR[i]==0): z[i] = -1
        else: z[i] = 1
    
    kf = (numpy.dot(DTR.T, DTR)+c)**d+ K**2
    zi_zj = numpy.dot(utils.vcol(z), utils.vrow(z))
    Hij = zi_zj * kf
    return z,Hij


# --- Radial Basis Function kernel SVM ---
def RBF (x1, x2, gamma, K):
    return numpy.exp(-gamma*(numpy.linalg.norm(x1-x2)**2)) + K**2


# --- ___ ---
def kernel_RBF_H (DTR, LTR, K, gamma):
    z=numpy.zeros(len(LTR))
    for i in range(len(LTR)):
        if(LTR[i]==0): z[i] = -1
        else: z[i] = 1
    
    kf = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kf[i,j] = RBF(DTR[:, i], DTR[:, j], gamma, K)

    zi_zj = numpy.dot(utils.vcol(z), utils.vrow(z))
    Hij = zi_zj * kf
    return z, Hij


# --- Training linear SVM ---
def SVM_linear_training (D, L, K_folds, K, C, z_value, pca_value=8):
    Dsplits, Lsplits = (numpy.array_split(D, K_folds, axis=1)), (numpy.array_split(L, K_folds))

    wrongs = 0
    llrs = []
    
    for i in range(K_folds):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        if z_value!=0: dtr, dts = utils.compute_znorm(dtr, dts)
        if pca_value!=0:
            P = utils.apply_PCA(dtr, pca_value)
            dtr, dts = numpy.dot(P.T,dtr), numpy.dot(P.T,dts)

        D,z,H = compute_mod_H(dtr, ltr, K)
        alpha_opt,_= minimize_mod_dual(D, C, z, H)
        _,wrong,llr = primal_model(D, alpha_opt, z, K, C, dts, lts)
        wrongs += wrong
        llrs.append(llr)

    for pi_value in pi_values:
        error_rate = wrongs / (D.shape[1])
        minDCF = costs.compute_min_DCF(pi_value, Cfn, Cfp, numpy.hstack(llrs), L)
        print(f"Linear SVM,{pca_value},{z_value},_,_,{C},_,_,_,{error_rate},{minDCF}\n")
    
    return llrs
    

# --- Training kernel quadratic SVM ---
def SVM_kernel_quadratic_training (D, L, K_folds, K, C, z_value, pca_value=8):
    c = 1
    d = 2
    Dsplits, Lsplits = (numpy.array_split(D, K_folds, axis=1)), (numpy.array_split(L, K_folds))

    wrongs = 0
    llrs = []
    
    for i in range(K_folds):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        if z_value!=0: dtr, dts = utils.compute_znorm(dtr, dts)
        if pca_value!=0:
            P = utils.apply_PCA(dtr, pca_value)
            dtr, dts = numpy.dot(P.T,dtr), numpy.dot(P.T,dts)
            
        z,H=kernel_poly_H(dtr, ltr, c, d, K)
        alpha_opt,_ = minimize_mod_dual(dtr, C, z, H)
        llr, wrong = scores_error_kernel_poly(alpha_opt, z, dtr, dts, lts, c, d, K)
        
        wrongs += wrong
        llrs.append(llr)
        
    error_rate = wrongs / (D.shape[1])
    minDCF = costs.compute_min_DCF(pi_eff, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Kernel Quadratic SVM,{pca_value},{z_value},_,_,{C},_,_,_,{error_rate},{minDCF}\n")
    
    return llrs


# --- Evaluating linear SVM ---
def SVM_linear_evaluation (DTR, LTR, DTE, LTE, K, C, z_value, pca_value=8):
    llrs = []
    
    if z_value!=0:
        DTR, DTE = utils.compute_znorm(DTR, DTE)
    if pca_value!=0:
        P = utils.apply_PCA(DTR, pca_value)
        DTR, DTE = numpy.dot(P.T,DTR), numpy.dot(P.T,DTE)
    
    D,z,H = compute_mod_H(DTR, LTR, K)
    alpha_opt,_ = minimize_mod_dual(D, C, z, H)
    _,_,llr = primal_model(D, alpha_opt, z, K, C, DTE, LTE)
    llrs.append(llr)
    
    return llrs


# --- Training radial quadratic SVM ---
def SVM_RBF_quadratic_training (D, L, K_folds, K, C, z_value, pca_value=8):
    gamma=0.001
    Dsplits, Lsplits = (numpy.array_split(D, K_folds, axis=1)), (numpy.array_split(L, K_folds))

    wrongs = 0
    llrs = []
    
    for i in range(K_folds):
        dtr, ltr = numpy.hstack(Dsplits[:i]+Dsplits[i+1:]), numpy.hstack(Lsplits[:i]+Lsplits[i+1:])
        dts, lts = numpy.asarray(Dsplits[i]), numpy.asarray(Lsplits[i])
        
        if z_value!=0: dtr, dts = utils.compute_znorm(dtr, dts)
        if pca_value!=0:
            P = utils.apply_PCA(dtr, pca_value)
            dtr, dts = numpy.dot(P.T,dtr), numpy.dot(P.T,dts)
           
        z,H = kernel_RBF_H(dtr, ltr, K, gamma)
        alpha_opt,_ = minimize_mod_dual_balanced(dtr, ltr, C, z, H, effective_prior)
        llr,wrong = scores_error_kernel_RBF(alpha_opt, z, dtr, dts, lts, K, gamma)
        
        wrongs += wrong
        llrs.append(llr)
        
    error_rate = wrongs/(D.shape[1])
    minDCF = costs.compute_min_DCF(pi_eff, Cfn, Cfp, numpy.hstack(llrs), L)
    print(f"Radial Quadratic SVM,{pca_value},{z_value},_,_,{C},_,_,_,{error_rate},{minDCF}\n")

    return llrs


# --- Evaluating radial quadratic SVM ---
def SVM_RBF_quadratic_evaluation (DTR, LTR, DTE, LTE, K, C, z_value, pca_value=8):
    gamma=0.001
    llrs= []
       
    if z_value!=0: DTR, DTE = utils.compute_znorm(DTR, DTE)
    if pca_value!=0:
        P = utils.apply_PCA(DTR, pca_value)
        DTR, DTE = numpy.dot(P.T,DTR), numpy.dot(P.T,DTE)
       
    z,H = kernel_RBF_H(DTR, LTR, K, gamma)
  
    alpha_opt, opt_dual_value = minimize_mod_dual(DTR, C, z, H)
    llr, wrong = scores_error_kernel_RBF(alpha_opt, z, DTR, DTE, LTE, K, gamma)
    llrs.append(llr)
    
    return llrs

def SVM_kernel_quadratic_evaluation (DTR, LTR, DTE, LTE, z_value, pca_value=8):
    llrs=[]
    
    if z_value!=0: DTR, DTE = utils.compute_znorm(DTR, DTE)
    if pca_value!=0:
        P = utils.apply_PCA(DTR,pca_value)
        DTR, DTE = numpy.dot(P.T,DTR), numpy.dot(P.T,DTE)
    
    z,H = kernel_poly_H(DTR, LTR,1,2,1)
    alpha_opt,_ = minimize_mod_dual(DTR, 0.001, z, H)
    llr,_ = scores_error_kernel_poly(alpha_opt, z, DTR, DTE, LTE, 1, 2, 1)
    llrs.append(llr)
    effective_prior=(0.5*1)/(0.5*1+0.5*10)
    #print("act dcf",computeActualDCF(0.5, 1, 10, np.hstack(svm_scores), LTE, False)) 
    print("min dcf eff",costs.compute_min_DCF(effective_prior, 1, 10, numpy.hstack(llrs), LTE)) 
    print("min dcf 0.5",costs.compute_min_DCF(0.5, 1, 10, numpy.hstack(llrs), LTE)) 
    print("min dcf 0.1",costs.compute_min_DCF(0.1, 1, 10, numpy.hstack(llrs), LTE)) 
    print("min dcf 0.9",costs.compute_min_DCF(0.9, 1, 10, numpy.hstack(llrs), LTE)) 

    return llrs