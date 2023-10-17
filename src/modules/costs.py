################################
#     MODULES IMPORTATIONS     #
################################
import numpy as np


####################
#    FUNCTIONS     #
####################
# ----- Compute unnormalized DCF -----
def compute_unnormalized_DCF (pi_t, Cfn, Cfp, ll_ratios, LTE, threshold=None, is_effective=False):
    # Define threshold if not previously defined
    if threshold is None:
        if is_effective is False:
            effective_prior = (pi_t*Cfn) / (pi_t*Cfn + (1-pi_t)*Cfp)
        else:
            effective_prior = pi_t
        threshold = -np.log(effective_prior / (1-effective_prior))
    
    # Build confusion matrix
    predicted = (ll_ratios>threshold).astype(int)
    confusion_matrix = np.zeros((2,2))
    for i in range(len(LTE)):
        confusion_matrix[predicted[i], LTE[i].astype(int)] += 1
    
    # Compute DCF
    fnr = confusion_matrix[0][1] / (confusion_matrix[0][1]+confusion_matrix[1][1])
    fpr = confusion_matrix[1][0] / (confusion_matrix[0][0]+confusion_matrix[1][0])
    dcf = pi_t*Cfn*fnr + (1-pi_t)*Cfp*fpr

    return dcf


# ----- Compute normalized DCF -----
def compute_normalized_DCF (pi_t, Cfn, Cfp, DCFu):
    dummy_costs = np.array([pi_t*Cfn, (1-pi_t)*Cfp])
    index = np.argmin(dummy_costs)
    dcf = DCFu / dummy_costs[index]
    return dcf


# ----- Compute actual DCF -----
def compute_actual_DCF (pi_t, Cfn, Cfp, ll_ratios, LTE, is_effective):
    dcfu = compute_unnormalized_DCF(pi_t, Cfn, Cfp, ll_ratios, LTE, None, is_effective)
    dcf = compute_normalized_DCF(pi_t, Cfn, Cfp, dcfu)
    return dcf


# ----- Compute minimum DCF -----
def compute_min_DCF (pi_t, Cfn, Cfp, ll_ratios, LTE):
    dcf_collection = np.zeros(ll_ratios.shape)
    sorted = np.sort(ll_ratios)

    for i in range(len(ll_ratios)):
        threshold = sorted[i]
        unnormalized = compute_unnormalized_DCF(pi_t, Cfn, Cfp, ll_ratios, LTE, threshold)
        dcf_collection[i] = compute_normalized_DCF(pi_t, Cfn, Cfp, unnormalized)
    return np.min(dcf_collection)