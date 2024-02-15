##### LIBRARIES #####
import numpy
import scipy
import utils as utils
import costs as costs
import math


##### GLOBAL VARIABLES #####
pi_eff = 0.5
Cfn = Cfn_eff = 1
Cfp = Cfp_eff = 10
effective_prior = (pi_eff*Cfn_eff) / (pi_eff*Cfn_eff + (1-pi_eff)*Cfp_eff)
pi_values = [0.1, 0.5, 0.9, effective_prior]


##### FUNCTIONS #####
# --- ___ ---
def GMM_logpdf_GAU_ND (X, mu, C):
    n_rows, n_cols = X.shape[0], X.shape[1]
    Y = numpy.empty([1, n_cols])
    
    for i in range(n_cols):
        x = X[:,i:i+1]
        _,det = numpy.linalg.slogdet(C)
        inv = numpy.linalg.inv(C)
        diff = x - mu
        Y[:,i] = -(n_rows/2)*numpy.log(2*math.pi) - 0.5*det - 0.5*numpy.dot(diff.T, numpy.dot(inv,diff))
    
    return Y.ravel()


# --- ___ ---
def logpdf_GMM (x, GMMs):
    y = []
    for w,mu,C in GMMs:
        lc = GMM_logpdf_GAU_ND(x, mu, C) + numpy.log(w)
        y.append(utils.vrow(lc))
    S = numpy.vstack(y)
    ld = scipy.special.logsumexp(y, axis=0)
    return S,ld


# --- ___ ---
def ML_GMM_LBG (D, w, mu, sigma, G, diagCov=False, tied=False):
    gmm = [(w,mu,sigma)]
    
    while len(gmm)<=G:
        if len(gmm)!=1: gmm = ML_GMM_IT(D,gmm, diagCov, 10, tied)
        if len(gmm)==G: break
        gmmNew=[]

        for (w,mu,C) in gmm:
            U,s,_ = numpy.linalg.svd(C)
            s[s<0.01] = 0.01
            C = numpy.dot(U, utils.vcol(s)*U.T)
            
            gmmNew.append((w*0.5, mu+s[0]**0.5*U[:, 0:1]*0.1, C)) 
            gmmNew.append((w*0.5, mu-s[0]**0.5*U[:, 0:1]*0.1, C))
        gmm = gmmNew
        
    return gmm
 

# --- ___ ---
def ML_GMM_IT (D, gmm, diagCov=False, nEMIters=10, tiedCov=False):
    new_ll = old_ll = None
    delta = 1.0
    _i = 0
    
    while delta>1e-6:
        lLL = []
        for w,mu,C in gmm:
            ll = GMM_logpdf_GAU_ND(D, mu, C) + numpy.log(w)
            lLL.append(utils.vrow(ll))
        LL = numpy.vstack(lLL)
        post = LL - scipy.special.logsumexp(LL, axis=0)
        post = numpy.exp(post)
        old_ll = new_ll
        new_ll = scipy.special.logsumexp(LL, axis=0).sum() / D.shape[1]
        
        if old_ll is not None: delta = new_ll - old_ll
        _i += 1
        psi = 0.01
        gmmUpd = []
        for i in range(post.shape[0]):
            Z = post[i].sum()
            F = (post[i:i+1, :]*D).sum(1)
            S = numpy.dot((post[i:i+1, :])*D, D.T)
            w_update = Z / D.shape[1]
            mu_update = utils.vcol(F/Z)
            C_update = S/Z - numpy.dot(mu_update, mu_update.T)
            
            if tiedCov:
                c=0
                for j in range(post.shape[0]):
                    Z = post[j].sum()
                    F = (post[j:j+1, :]*D).sum(1)
                    S = numpy.dot((post[j:j+1, :])*D, D.T)
                    c += Z * (S/Z - numpy.dot(utils.vcol(F/Z), utils.vcol(F/Z).T))
                C_update = c / D.shape[1]
            
            if diagCov: C_update *= numpy.eye(C_update.shape[0])
            
            U,s,_ = numpy.linalg.svd(C_update)
            s[s<psi] = psi
            C_update = numpy.dot(U, utils.vcol(s)*U.T)
            gmmUpd.append((w_update, mu_update, C_update))

        gmm = gmmUpd
        lLL = []
        for w,mu,C in gmm:
            ll = GMM_logpdf_GAU_ND(D, mu, C) + numpy.log(w)
            lLL.append(utils.vrow(ll))
        LL = numpy.vstack(lLL)
        post = LL - scipy.special.logsumexp(LL, axis=0)
        post = numpy.exp(post)
        old_ll = new_ll
        new_ll = scipy.special.logsumexp(LL, axis=0).sum() / D.shape[1]
        delta = new_ll - old_ll
            
    return gmm


# --- Training GMM ---
def GMM_train (D, L, K_folds, g0_value, g1_value, z_value, diag, tied, pca_value=8):
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
            
        DTR0 = dtr[:, ltr==0]
        DTR1 = dtr[:, ltr==1]

        w0 = 1.0
        mu0 = DTR0.mean(1).reshape((DTR0.shape[0], 1))
        diff = DTR0 - mu0
        C0 = 1 / (DTR0.shape[1]) * numpy.dot(diff,diff.T)
        if diag: C0 *= numpy.eye(C0.shape[0])
        U,s,_ = numpy.linalg.svd(C0)
        s[s<0.01] = 0.01
        C0 = numpy.dot(U, utils.vcol(s)*U.T)
        
        uGMM0 = ML_GMM_LBG(DTR0, w0, mu0, C0, g0_value, diag, tied)
        _,score0 = logpdf_GMM(dts, uGMM0)
       
        
        w1 = 1.0
        mu1 = DTR1.mean(1).reshape((DTR1.shape[0], 1))
        diff = DTR1 - mu1
        C1 = 1 / (DTR1.shape[1]) * numpy.dot(diff, diff.T)
        if diag: C1 *= numpy.eye(C1.shape[0])
        U,s,_ = numpy.linalg.svd(C1)
        s[s<0.01] = 0.01
        C1 = numpy.dot(U, utils.vcol(s)*U.T)
        
        uGMM1 = ML_GMM_LBG(DTR1, w1, mu1, C1, g1_value, diag, tied)
        _,score1 = logpdf_GMM(dts, uGMM1)

        S_joint = numpy.vstack((score0,score1))
        S_marg = utils.vrow(scipy.special.logsumexp(S_joint, axis=0))
        f = numpy.exp(S_joint - S_marg)
        
        wrongs += (f.argmax(0)!=lts).sum()
        llrs.append((score1-score0)[0])
    
    error_rate = wrongs/(D.shape[1])
    minDCF = costs.compute_min_DCF(pi_eff, Cfn, Cfp, numpy.hstack(llrs), L)    
    if tied:
        if diag: print(f"GMM Tied Diagonal Covariance,{pca_value},{z_value},_,_,_,_,{g0_value},{g1_value},{error_rate},{minDCF}\n")
        else: print(f"GMM Tied Covariance,{pca_value},{z_value},_,_,_,_,{g0_value},{g1_value},{error_rate},{minDCF}\n")
    else:
        if diag: print(f"GMM Diagonal Covariance,{pca_value},{z_value},_,_,_,_,{g0_value},{g1_value},{error_rate},{minDCF}\n")
        else: print(f"GMM Full Covariance,{pca_value},{z_value},_,_,_,_,{g0_value},{g1_value},{error_rate},{minDCF}\n")
    
    return llrs


# --- Evaluating GMM ---
def GMM_eval (DTR, LTR, DTE, LTE, g0_value, g1_value, z_value, diag, tied, pca_value=8):
        gmm_scores=[]
        
        if z_value!=0: DTR, DTE = utils.znorm(DTR, DTE)
        if pca_value!=0:
            P = utils.apply_PCA(DTR, pca_value)
            DTR, DTR = numpy.dot(P.T,DTR), numpy.dot(P.T,DTE)
            
        DTR0 = DTR[:,LTR==0]
        DTR1 = DTR[:,LTR==1]
        
        w0 = 1.0
        mu0 = DTR0.mean(1).reshape((DTR0.shape[0], 1))
        C0 = 1 / (DTR0.shape[1]) * numpy.dot((DTR0-mu0), (DTR0-mu0).T)
        if diag: C0 *= numpy.eye(C0.shape[0])
        U,s,_ = numpy.linalg.svd(C0)
        s[s<0.01] = 0.01
        C0 = numpy.dot(U, utils.vcol(s)*U.T)
        
        uGMM0 = ML_GMM_LBG(DTR0, w0, mu0, C0, g0_value, diag, tied)
        _,score0=logpdf_GMM(DTE, uGMM0)
        
        w1 = 1.0
        mu1 = DTR1.mean(1).reshape((DTR1.shape[0], 1))
        C1 = 1 / (DTR1.shape[1]) * numpy.dot((DTR1-mu1), (DTR1-mu1).T)
        if diag: C1 *= numpy.eye(C1.shape[0])
        U, s, _ = numpy.linalg.svd(C1)
        s[s<0.01] = 0.01
        C1 = numpy.dot(U, utils.vcol(s)*U.T)
        
        uGMM1 = ML_GMM_LBG(DTR1, w1, mu1, C1, g1_value, diag, tied)
        _,score1 = logpdf_GMM(DTE, uGMM1)
        
        gmm_scores.append((score1-score0)[0])
        
        return numpy.hstack(gmm_scores)