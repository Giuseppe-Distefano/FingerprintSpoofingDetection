##### LIBRARIES #####
import numpy


##### GLOBAL VARIABLES #####
pi_eff = 0.5
Cfn_eff = 1
Cfp_eff = 10
effective_prior = (pi_eff*Cfn_eff) / (pi_eff*Cfn_eff + (1-pi_eff)*Cfp_eff)


##### FUNCTIONS #####
# --- Compute DCF ---
def compute_DCF (pi_value, Cfn, Cfp, llrs, LTE, threshold=None, effective=False):
    if threshold is None:
        if effective is False:
            effective_prior=(pi_value*Cfn)/(pi_value*Cfn + (1-pi_value)*Cfp)
        else:
            effective_prior=pi_value
        threshold = -numpy.log((effective_prior)/(1-effective_prior))


    predictedLabels = (llrs > threshold).astype(int)

    confMatrix = numpy.zeros((2, 2))
    
    for i in range(len(LTE)):
        confMatrix[predictedLabels[i], LTE[i].astype(int)] += 1
    FNR = confMatrix[0][1]/(confMatrix[0][1]+confMatrix[1][1])
    FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])

    DCFu = pi_value*Cfn*FNR+(1-pi_value)*Cfp*FPR

    return DCFu


# --- Compute normalized DCF ---
def compute_normalized_DCF (pi_value, Cfn, Cfp, DCFu):
    dummy_costs = numpy.array([pi_value*Cfn, (1-pi_value)*Cfp])
    r = numpy.argmin(dummy_costs)
    nDCF = DCFu/dummy_costs[r]
    return nDCF


# --- Compute actual DCF ---
def compute_actual_DCF (pi_value, Cfn, Cfp, llrs, LTE, effective):
    dcfu=compute_DCF(pi_value,Cfn,Cfp,llrs,LTE,None,effective)
    aDCF=compute_normalized_DCF(pi_value,Cfn,Cfp,dcfu)
    return aDCF

 
# --- Compute min DCF ---
def compute_min_DCF(pi_value, Cfn, Cfp, llrs, LTE):
    DCF_collection = numpy.zeros(llrs.shape)
    sorted_ratios = numpy.sort(llrs)

    for i in range(len(llrs)):
       threshold = sorted_ratios[i]
       DCFu = compute_DCF(pi_value, Cfn, Cfp, llrs, LTE, threshold)
       DCF_collection[i] = compute_normalized_DCF(pi_value, Cfn, Cfp, DCFu)

    return numpy.min(DCF_collection)
