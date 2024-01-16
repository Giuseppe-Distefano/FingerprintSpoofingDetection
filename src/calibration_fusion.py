################################
#     MODULES IMPORTATIONS     #
################################
import numpy
import matplotlib.pyplot as plt
import utility as utility
import discriminative_models as disc
import optimal_costs as dcf


###########################
#     GLOBAL VARIABLES    #
###########################
pi_t = 0.5
Cfn = 1
Cfp = 10
effective_prior = (pi_t*Cfn) / (pi_t*Cfn + (1-pi_t)*Cfp)
#output_folder = "../output/Calibration_Fusion/"


####################
#     FUNCTIONS    #
####################
# ----- Bayes Error plot -----
def bayes_error_plot (ll_ratios, labels):
    title="title"
    eff_plo = numpy.linspace(-4, 4, 10)
    eff_p = 1 / (1 + numpy.exp(-1*eff_plo))
    actDCF=[]
    minDCF=[]

    for i in range(10):
        print("iteration ", i)
        actDCF.append(dcf.computeActualDCF(eff_p[i], 1, 1, ll_ratios, labels, True))
        minDCF.append(dcf.computeMinDCF(eff_p[i], 1, 1, ll_ratios, labels))
        #actdcf.append(computeActualDCF(effPriors[i], Cfn, Cfp, llrs, labels, True))
        #mindcf.append(computeMinDCF(effPriors[i], Cfn, Cfp, llrs, labels))

    plt.plot(eff_plo, actDCF, label='Actual DCF', color="y")
    plt.plot(eff_plo, minDCF, label='Min DCF', color="r", linestyle="--")
    plt.xlabel("log pi/(1-pi)") 
    plt.ylabel("DCF value")
    plt.legend(["act DCF", "min DCF"])
    plt.title("Bayes Error Plot"+ title)
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.savefig('./images/bayes_plot_' + title+ '.png')
    plt.show()    
    return
    plt.plot(eff_plo, actDCF, label='Actual DCF', color="b")
    plt.plot(eff_plo, minDCF, label='Min DCF', color="r", linestyle="--")
    plt.xlabel("Effective prior log-odds") 
    plt.ylabel("DCF value")
    plt.legend(["act DCF", "min DCF"])
    plt.title("Bayes Error Plot"+ title)
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    #plt.savefig(output_folder + title+ '.png')


# ----- Score calibration -----
    
def score_calibration (DTE, v, pi_value=effective_prior):
    s = []
    w,b = v[0:-1], v[-1]
    for i in range(DTE.shape[1]):
        xt = DTE[:,i]
        s.append(b + numpy.dot(w.T, xt) - numpy.log((pi_value) / (1-pi_value)))
    return numpy.array(s)
    
"""
def score_calibration(DTE,v, eff_prior):
    
    s=numpy.empty((DTE.shape[1]))
    LP=numpy.empty((DTE.shape[1]))
    
    w, b = v[0:-1], v[-1]
    
    for i in range(DTE.shape[1]):
        #s[i]=np.dot(w.T,DTE[:,i])+b
        s[i]=numpy.dot(w.T,DTE[:,i])+b-numpy.log(eff_prior/(1-eff_prior))
    return s"""


# ----- Class calibration -----
def class_calibration (DTR, LTR, DTE, LTE, pi_value, lambda_value):
    x0 = numpy.zeros(DTR.shape[0] + 1)
    J = disc.weighted_logreg_obj_wrap(DTR, LTR, lambda_value, pi_value)
    grad = disc.gradient_test(DTR, LTR, lambda_value, pi_value)
    x,_ = disc.numerical_optimization(J, x0, grad)
    print("x",x)
    scores = score_calibration(DTE, x, pi_value)
    #print("Scores nostri",scores)
    predicted_labels = utility.define_predictions(scores, 0)
    wrong_predictions = utility.count_wrong_prediction(predicted_labels, LTE)
    print("wrongPredictions,scores",wrong_predictions,scores)
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
        labels = numpy.append(labels,LTE)

    print("llr",len(ll_ratios))
    print("NOSTRA: Linear Logistic Regression minDCF")
    print(dcf.computeMinDCF(0.5, 1, 10, numpy.hstack(ll_ratios), labels))  
    bayes_error_plot(numpy.hstack(ll_ratios), labels)

    return ll_ratios


def calibrate_scores(s, L):
    s = numpy.hstack(s)
    #s_rand, L_rand = utility.randomize(s.reshape(1,s.size),L,0)
    print("S",s[0:5])
    numpy.random.seed(100) 
    indexes = numpy.random.permutation(s.shape[0])
    s_rand = numpy.zeros((1, s.shape[0]))
    L_rand = numpy.zeros((L.size,))

    index = 0
    for rand_index in indexes:
        s_rand[0,index] = s[rand_index]
        L_rand[index] = L[rand_index]
        index+=1
    

    
    effective_prior=(0.5*1)/(0.5*1+0.5*10)
    
    #print(s_rand[0,0:5])
    #calibrated_scores = dm.linear_lr_train_calibration(s_rand, L_rand,2,0.0001,effective_prior)
    calibrated_scores = compute_calibration(s_rand, L_rand, 2,effective_prior, 1e-4 )

    print("calibrated_scores", numpy.hstack(calibrated_scores)[0:5])
    actualDCF = dcf.computeActualDCF(0.5, 1, 10, numpy.hstack(calibrated_scores), L_rand,False)
    #actualDCF = dcf.computeActualDCF(pi_t, Cfn, Cfp, calibrated_scores, numpy.hstack(L_rand), False)

    print(dcf.computeMinDCF(0.5, 1, 10, numpy.hstack(calibrated_scores), L_rand))
    #print("calibrated_scores",calibrated_scores)
    print("act dcf", actualDCF)


    #bayes_error_plot(s, L)
    #bayes_error_plot(numpy.hstack(calibrated_scores), numpy.hstack(L_rand))

    return calibrated_scores, L_rand

# ----- Calibrate scores -----
"""def calibrate_scores (scores, labels):
    scores = numpy.hstack(scores)
    numpy.random.seed(100)
    indexes = numpy.random.permutation(scores.shape[0])
    sc = numpy.zeros((1, scores.shape[0]))
    lab = numpy.zeros((labels.size,))
    i = 0
    for ind in indexes:
        sc[0,i] = sc[ind]
        lab[i] = labels[ind]
        i += 1

    calibrated = compute_calibration(sc, lab, 2, effective_prior, 1e-4)
    #actualDCF = dcf.computeActualDCF(pi_t, Cfn, Cfp, numpy.hstack(calibrated), numpy.hstack(lab), False)

    bayes_error_plot(scores, labels)
    bayes_error_plot(numpy.hstack(calibrated), numpy.hstack(lab))

    return calibrated"""


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
        sc[:,i] = sc[:,ind]
        lab[i] = labels[ind]
        i += 1

    calibrated = compute_calibration(sc, lab, 2, effective_prior)
    #actualDCF = dcf.computeActualDCF(pi_t, Cfn, Cfp, numpy.hstack(calibrated), numpy.hstack(lab), False)

    bayes_error_plot(numpy.hstack(calibrated), numpy.hstack(lab))

    return calibrated
