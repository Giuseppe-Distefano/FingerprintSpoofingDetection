##### MODULES IMPORTATIONS #####
import utility as u
import matplotlib.pyplot as plt
import plot 
import PCA_LDA as dimRed
##### CLASSES #####


##### FUNCTIONS #####


##### MAIN OF THE PROGRAM #####
if __name__ == "__main__":
    DTR, LTR =u.load(".\dataset\Train.txt")
    DTE, LTE =u.load(".\dataset\Test.txt")

    mu = u.computeMean(DTR)
    C = u.computeCovarianceMatrix(DTR)
    m = 2 #dim 2
    DP = dimRed.computeEigenValuesAndEigenVectors(C,DTR,m)
    plot.plot_scatter_PCA(DP,LTR)
    DP = dimRed.compute_LDA_directions1(DTR,LTR,m)
    plot.plot_hist_LDA(DP,LTR)


