################################
#     MODULES IMPORTATIONS     #
################################
import numpy
import matplotlib.pyplot as plt
import modules.utility as utility
import modules.discriminative as disc
import modules.costs as dcf


###########################
#     GLOBAL VARIABLES    #
###########################
pi_t = 0.5
Cfn = 1
Cfp = 10
effective_prior = (pi_t*Cfn) / (pi_t*Cfn + (1-pi_t)*Cfp)
output_folder = "../output/Calibration_Fusion/"


####################
#     FUNCTIONS    #
####################
# ----- Bayes Error plot -----
def bayes_error_plot (ll_ratios, labels, title):
    eff_plo = numpy.linspace(-4, 4, 10)
    eff_p = 1 / (1 + numpy.exp(-1*eff_plo))
    actDCF = minDCF = []

    for i in range(10):
        actDCF.append(dcf.compute_actual_DCF(eff_p[i], 1, 1, numpy.hstack(ll_ratios), numpy.hstack(labels), True))
        minDCF.append(dcf.compute_min_DCF(eff_p[i], 1, 1, numpy.hstack(ll_ratios), numpy.hstack(labels)))

    plt.plot(eff_plo, actDCF, label='Actual DCF', color="b")
    plt.plot(eff_plo, minDCF, label='Min DCF', color="r", linestyle="--")
    plt.xlabel("Effective prior log-odds") 
    plt.ylabel("DCF value")
    plt.legend(["act DCF", "min DCF"])
    plt.title("Bayes Error Plot"+ title)
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.savefig(output_folder + title+ '.png')


# ----- Score calibration -----
def score_calibration (DTE, v, pi_value=effective_prior):
    s = numpy.empty((DTE.shape[1]))
    w,b = v[0:-1], v[-1]
    for i in range(DTE.shape[1]):
        xt = DTE[:,i]
        s[i] = b + numpy.dot(w.T, xt) - numpy.log((pi_value) / (1-pi_value))
    return numpy.array(s)


# ----- Class calibration -----
def class_calibration (DTR, LTR, DTE, LTE, pi_value, lambda_value):
    x0 = numpy.zeros(DTR.shape[0] + 1)
    x = disc.numerical_optimization(disc.lr_obj_wrap(DTR, LTR, lambda_value, pi_value), x0, disc.lr_compute_gradient(DTR, LTR, lambda_value, pi_value))

    scores = score_calibration(DTE, x, pi_value)
    predicted_labels = utility.predict_labels(scores, 0)
    wrong_predictions = utility.count_mispredictions(predicted_labels, LTE)

    return wrong_predictions, scores


# ----- Calibration function -----
def compute_calibration (D, L, K, pi_value, lambda_value):
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

        # Apply classifier
        wrong, scores = class_calibration(DTR, LTR, DTE, LTE, pi_value, lambda_value)
        wrong_predictions += wrong
        ll_ratios.append(scores)
        labels.append(LTE)

    return numpy.hstack(ll_ratios)


# ----- Calibrate scores -----
def calibrate_scores (scores, labels, model_title):
    scores = numpy.hstack(scores)

    numpy.random.seed(100)
    indexes = numpy.random.permutation(scores.shape[0])
    sc = numpy.zeros((1, scores.shape[0]))
    lab = numpy.zeros((labels.size,))
    i = 0
    for ind in indexes:
        sc[0,i] = scores[ind]
        lab[i] = labels[ind]
        i += 1

    calibrated = compute_calibration(sc, lab, 2, effective_prior, 1e-4)
    
    bayes_error_plot(scores, labels, model_title+'_uncalibrated')
    bayes_error_plot(numpy.hstack(calibrated), lab, model_title+'_calibrated')

    return calibrated


# ----- Fuse two models -----
def fuse_models (model1, scores1, model2, scores2, labels):
    s = [ numpy.hstack(scores1), numpy.hstack(scores2) ]
    s = numpy.vstack(s)
    numpy.random.seed(5)
    indexes = numpy.random.permutation(s.shape[1])
    sc = numpy.zeros((2, s.shape[1]))
    lab = numpy.zeros((labels.size,))
    i = 0
    for ind in indexes:
        sc[:,i] = s[:,ind]
        lab[i] = labels[ind]
        i += 1

    calibrated = compute_calibration(sc, lab, 2, effective_prior, 1e-3)
    
    bayes_error_plot(numpy.hstack(calibrated), numpy.hstack(lab), model1+'_'+model2)

    return calibrated
