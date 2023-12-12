################################
#     MODULES IMPORTATIONS     #
################################
import modules.utility as utility
import numpy
import scipy.optimize
import modules.costs as dcf
import modules.pca_lda as dr




####################
#     FUNCTIONS    #
####################
# ----- ___ -----

#computes the log-density of a GMM for a set of samples contained in matrix X
def logpdf_GMM(x, gmm):
    
    y=[]
    for weight, mu,sigma in gmm:
        lc=utility.logpdf_GAU_ND(x, mu, sigma)+ numpy.log(weight)
        y.append(utility.vrow(lc))
    S=numpy.vstack(y)
    logdens=scipy.special.logsumexp(y, axis=0)
    return S,logdens


def ML_GMM_LBG(D, weights, means, sigma, num_components, diagCov=False, tied=False):

    gmm = [(weights,means,sigma)]
    newGMM=[]

    while len(gmm)<=num_components:
       
        if len(gmm)!=1: 
            gmm = ML_GMM_iteration(D,gmm, diagCov, 10, tied)
            #print("iterazione gmm", gmm)

        if len(gmm)==num_components: 
            break

        #newGMM=[]
        for(weight, mu, sigma) in gmm:
            U,s,_=numpy.linalg.svd(sigma)
            s[s<0.01] = 0.01
            sigma= numpy.dot(U, utility.vcol(s)*U.T)
            
            newGMM.append((weight*0.5, mu+s[0]**0.5*U[:, 0:1]*0.1,sigma )) 
            newGMM.append((weight*0.5, mu-s[0]**0.5*U[:, 0:1]*0.1,sigma ))

        #gmm = newGMM
        
    #return gmm
    return newGMM
       
def ML_GMM_iteration(D, gmm, diagCov=False, num_em_iters=10, tiedCov=False):
    prevLL=None
    oldLL=None
    deltaLL=1.0
    iteration=0
    while deltaLL>1e-6:
        #print("Delta", deltaLL)
        componentsLL=[]

        for weight, mean, sigma in gmm:
            ll=utility.logpdf_GAU_ND(D, mean, sigma)+numpy.log(weight)
            componentsLL.append(utility.vrow(ll))
        LL=numpy.vstack(componentsLL) #S

        posterior =LL-scipy.special.logsumexp(LL, axis=0) #logdens
        posterior =numpy.exp(posterior )
        oldLL=LL
        prevLL=scipy.special.logsumexp(LL, axis=0).sum()/D.shape[1]
        
        if oldLL is not None:
            deltaLL=prevLL-oldLL
            #print(_ll)
        iteration=iteration+1
        psi=0.01
        updatedGMM=[]
        for i in range(post.shape[0]):
            Z=posterior[i].sum()
            F=(posterior[i:i+1, :]*D).sum(1)
            S=numpy.dot((post[i:i+1, :])*D, D.T)
            newWeight=Z/D.shape[1]
            #print(newWeight)
            newMu=utility.vcol(F/Z)
            newCov=S/Z - numpy.dot(newMu, newMu.T)
            
            if tiedCov:
                c=0
                for j in range(post.shape[0]):
                    Z=post[j].sum()
                    F=(post[j:j+1, :]*D).sum(1)
                    S=numpy.dot((post[j:j+1, :])*D, D.T)
                    c+=Z*(S/Z-numpy.dot(utility.vcol(F/Z), utility.vcol(F/Z).T))
                newCov=1/D.shape[1]*c
            
            if diagCov:
                newCov=newCov*numpy.eye(newCov.shape[0])
            
            U, s, _=numpy.linalg.svd(newCov)
            s[s<psi]=psi
            newCov=numpy.dot(U, utility.vcol(s)*U.T)
            updatedGMM.append((newWeight, newMu, newCov))
            
                
        gmm=updatedGMM
        lLL=[]
        for w, mu, C in gmm:
            ll=utility.logpdf_GAU_ND(D, mu, C)+numpy.log(w)
            lLL.append(utility.vrow(ll))
        LL=numpy.vstack(lLL) #S
        post=LL-scipy.special.logsumexp(LL, axis=0) #logdens
        post=numpy.exp(post)
        oldLL=prevLL
        prevLL=scipy.special.logsumexp(LL, axis=0).sum()/D.shape[1]
        deltaLL=prevLL-oldLL
            
    return gmm