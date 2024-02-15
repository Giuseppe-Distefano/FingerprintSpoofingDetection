##### LIBRARIES #####
import numpy
import costs 
import discriminative 


##### GLOBAL VARIABLES #####
pi_eff = 0.5
Cfn = Cfn_eff = 1
Cfp = Cfp_eff = 10
effective_prior = (pi_eff*Cfn_eff) / (pi_eff*Cfn_eff + (1-pi_eff)*Cfp_eff)

pca_values = [0, 9, 8, 7, 6, 5]
pi_values = [0.1, 0.5, 0.9]


##### FUNCTIONS #####
# --- Calibration ---
def calibrate_scores(s, L):
    s = numpy.hstack(s)
    numpy.random.seed(100) 
    indexes = numpy.random.permutation(s.shape[0])
    s_rand = numpy.zeros((1, s.shape[0]))
    L_rand = numpy.zeros((L.size,))
    i = 0
    for ind in indexes:
        s_rand[0,i] = s[ind]
        L_rand[i] = L[ind]
        i += 1
    
    calibrated = discriminative.linear_lr_train_calibration(s_rand, L_rand, 2, 1e-4, effective_prior)
    actualDCF = costs.compute_actual_DCF(pi_eff, Cfn_eff, Cfp_eff, numpy.hstack(calibrated), L_rand, False)
    
    return calibrated, actualDCF, L_rand


# --- Fusion ---
def fuse_scores (llr1, llr2, L, effective_prior=None):
    s = [numpy.hstack(llr1),numpy.hstack(llr2)]
    s_new=numpy.vstack(s)
    
    numpy.random.seed(5) 
    indexes = numpy.random.permutation(s_new.shape[1])
    s_rand = numpy.zeros((2, s_new.shape[1]))
    L_rand = numpy.zeros((L.size,))
    index = 0
    for rand_index in indexes:
        s_rand[:,index] = s_new[:,rand_index]
        L_rand[index] = L[rand_index]
        index+=1
        
    if effective_prior is None:
        effective_prior=(0.5*1)/(0.5*1+0.5*10)
    
    calibrated_scores = discriminative.linear_lr_train_calibration(s_rand, L_rand,2,0.001,effective_prior)
    actualDCF = costs.compute_actual_DCF(0.5, 1, 10, numpy.hstack(calibrated_scores), L_rand,False)
    
    return calibrated_scores, actualDCF, L_rand
