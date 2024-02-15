##### LIBRARIES #####
import numpy
import utils as utils
import pylab
import matplotlib.pyplot
import seaborn
import costs as costs
import calibration as calibration


##### FUNCTIONS #####
# --- Plot histograms of classes from dataset ---
def plot_histograms ():
    DTR, LTR = utils.load_data('Train.txt')
    
    for i in range(10):
        labels = ["0", "1"]
        title ="feature"+ str(i)
        matplotlib.pyplot.figure()
        matplotlib.pyplot.title(title)

        y = DTR[:, LTR==0][i]
        matplotlib.pyplot.hist(y, bins=40, density=True, alpha=0.4, linewidth=1.0, color='green', edgecolor='black', label=labels[0])
        y = DTR[:, LTR == 1][i]
        matplotlib.pyplot.hist(y, bins=40, density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black', label=labels[1])
        matplotlib.pyplot.legend()
        matplotlib.pyplot.savefig('./images/hist_' + title + '.jpg')
        matplotlib.pyplot.show()


# --- Plot scatters of samples from dataset ---
def plot_scatters ():
    DTR, LTR = utils.load_data('Train.txt')

    P = utils.apply_PCA(DTR, 2)
    D = numpy.dot(P.T,DTR)
    D0 = D[:, LTR==0]
    D1 = D[:, LTR==1]
    
    pylab.scatter(D0[0],D0[1],label="Spoofed Fingerprints")
    pylab.scatter(D1[0],D1[1],label="Authentic Fingerprints")
    pylab.legend()
    pylab.tight_layout()
    matplotlib.pyplot.savefig('./images/scatter_PCA2' + '.jpg')
    matplotlib.pyplot.show()        


# --- Plot heatmaps showing correlations between samples ---
def plot_correlations (DTR, title, cmap="Greys"):
    corr = numpy.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = utils.compute_correlation(X, Y)
            corr[x][y] = pearson_elem

    seaborn.set()
    heatmap = seaborn.heatmap(numpy.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False)
    fig = heatmap.get_figure()
    fig.savefig("./images/" + title + ".jpeg")


# --- ___ ---
def plot_roc_curve(FPR, TPR):
    # Function used to plot TPR(FPR)
    matplotlib.pyplot.figure()
    matplotlib.pyplot.grid(linestyle='--')
    matplotlib.pyplot.plot(FPR, TPR, linewidth=2)
    matplotlib.pyplot.xlabel("FPR") 
    matplotlib.pyplot.ylabel("TPR")
    matplotlib.pyplot.title("ROC curve")
    matplotlib.pyplot.show()


# --- Bayes error plots for evaluation ---
def bayesErrorPlot(llrs,labels, title): 
    effPriorLogOdds = numpy.linspace(-4, 4,10)
    effPriors=1/(1+numpy.exp(-1*effPriorLogOdds))
    Cfn = Cfp = 1
    actdcf=[]
    mindcf = []
    print("effPriors",effPriors.shape)
    print("effPriorLogOdds",effPriorLogOdds.shape)

    for i in range(10):
        print("iteration ", i)       
        actdcf.append(costs.compute_actual_DCF(effPriors[i], Cfn, Cfp, llrs, labels, True))
        mindcf.append(costs.compute_min_DCF(effPriors[i], Cfn, Cfp, llrs, labels))
        
    matplotlib.pyplot.plot(effPriorLogOdds, actdcf, label='Actual DCF', color="y")
    matplotlib.pyplot.plot(effPriorLogOdds, mindcf, label='Min DCF', color="r", linestyle="--")
    matplotlib.pyplot.xlabel("log pi/(1-pi)") 
    matplotlib.pyplot.ylabel("DCF value")
    matplotlib.pyplot.legend(["act DCF", "min DCF"])
    matplotlib.pyplot.title("Bayes Error Plot"+ title)
    matplotlib.pyplot.ylim([0, 1])
    matplotlib.pyplot.xlim([-4, 4])
    matplotlib.pyplot.savefig('./images/bayes_plot_' + title+ '.png')
    matplotlib.pyplot.show()


# --- ___ ---
def plotDCF_lambda(x, y, xlabel, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.plot(x, y[0:len(x)], label='min DCF noPCA noZNorm', color='b')
    matplotlib.pyplot.plot(x, y[len(x): 2*len(x)], label='min DCF noPCA ZNorm', color='r')
    matplotlib.pyplot.plot(x, y[2*len(x): 3*len(x)], label='min DCF PCA=8 NoZnorm', color='g')
    matplotlib.pyplot.plot(x, y[3*len(x): 4*len(x)], label='min DCF PCA=8 Znorm', color='y')
    matplotlib.pyplot.xlim([min(x), max(x)])
    matplotlib.pyplot.ylim([min(y), max(y)])

    matplotlib.pyplot.xscale("log")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(["min DCF noPCA noZNorm", "min DCF noPCA ZNorm", "min DCF PCA=8 NoZnorm",'min DCF PCA=8 Znorm'])
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel("min DCF")
    matplotlib.pyplot.savefig('./images/dcf_lamba_' + title + '.jpg')
    matplotlib.pyplot.show()


# --- ___ ---
def plotDCF_lambda_eval(x, y1, y2, xlabel, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.plot(x, y1, label='min DCF [eval set]', color='b')
    matplotlib.pyplot.plot(x, y2, label='min DCF [valid set]', color='b', linestyle="--")
    matplotlib.pyplot.xlim([min(x), max(x)])
    matplotlib.pyplot.xscale("log")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(['min DCF [eval set]','min DCF [valid set]'])
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel("min DCF")
    matplotlib.pyplot.savefig('./images/dcf_lamba' + title + '.jpg')
    matplotlib.pyplot.show()


# --- ___ ---
def plotDCF_C_eval(x, y1, y2, xlabel, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.plot(x, y1, label='min DCF [eval set]', color='b')
    matplotlib.pyplot.plot(x, y2, label='min DCF [valid set]', color='b', linestyle="--")
    matplotlib.pyplot.xlim([min(x), max(x)])
    matplotlib.pyplot.xscale("log")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(['min DCF [eval set]','min DCF [valid set]'])
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel("min DCF")
    matplotlib.pyplot.savefig('./images/dcf_c_poly ' + title+ '.jpg')
    matplotlib.pyplot.show()


# --- ___ ---
def plotDCF_gamma(x, y, xlabel, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.plot(x, y[0:len(x)], label='min DCF gamma=0.1', color='b')
    matplotlib.pyplot.plot(x, y[len(x): 2*len(x)], label='min DCF gamma=0.01', color='r')
    matplotlib.pyplot.plot(x, y[2*len(x): 3*len(x)], label='min DCFgamma=0.001', color='g')
    
    matplotlib.pyplot.xlim([min(x), max(x)])
    matplotlib.pyplot.xscale("log")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(["min DCF gamma=0.1", "min DCF gamma=0.01", "min DCF gamma=0.001"])
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel("min DCF")
    matplotlib.pyplot.savefig('./images/dcf_c_rbf' + title+ '.jpg')
    matplotlib.pyplot.show()


# --- ___ ---
def plotDCF_c(x, y, xlabel, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.plot(x, y[0:len(x)], label='min DCF c=0', color='b')
    matplotlib.pyplot.plot(x, y[len(x): 2*len(x)], label='min DCF c=1', color='r')
    matplotlib.pyplot.xlim([min(x), max(x)])
    matplotlib.pyplot.xscale("log")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(["min DCF c=0", "min DCF c=1"])
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel("min DCF")
    matplotlib.pyplot.savefig('./images/dcf_c_poly' + title+ '.jpg')
    matplotlib.pyplot.show()


# --- ___ ---
def plotDCF_C(x, y, xlabel, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.plot(x, y[0:len(x)], label='min DCF K=1 noPCA noZNorm', color='b')
    matplotlib.pyplot.plot(x, y[len(x): 2*len(x)], label='min DCF K=1 noPCA ZNorm', color='r')
    matplotlib.pyplot.plot(x, y[2*len(x): 3*len(x)], label='min DCF K=1 PCA=8 NoZnorm', color='g')
    matplotlib.pyplot.plot(x, y[3*len(x): 4*len(x)], label='min DCF K=1 PCA=8 Znorm', color='y')
    matplotlib.pyplot.xlim([min(x), max(x)])
    matplotlib.pyplot.xscale("log")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(["min DCF K=1 noPCA noZNorm", "min DCF K=1 noPCA ZNorm", "min DCF K=1 PCA=8 NoZnorm",'min DCF K=1 PCA=8 Znorm'])
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel("min DCF")
        
    matplotlib.pyplot.savefig('./images/dcf_C' + title +'.jpg')
    matplotlib.pyplot.show()


# --- ___ ---
def plotDCF_gmm_comp(x, y, xlabel, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.plot(x, y[0:len(x)], label='min DCF G0=1', color='b')
    matplotlib.pyplot.plot(x, y[len(x): 2*len(x)], label='min DCF G0=2', color='r')
    matplotlib.pyplot.plot(x, y[2*len(x): 3*len(x)], label='min DCF G0=4', color='g')
    matplotlib.pyplot.plot(x, y[3*len(x): 4*len(x)], label='min DCF G0=8', color='y')
    matplotlib.pyplot.xlim([min(x), max(x)])
    matplotlib.pyplot.xscale("linear")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(["min DCF G0=1", "min DCF G0=2", "min DCF G0=4", "min DCF G0=8"])
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel("min DCF")
    matplotlib.pyplot.savefig('./images/dcf_gmm_components' + title+ '.jpg')
