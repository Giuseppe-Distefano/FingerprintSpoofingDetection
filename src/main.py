##### MODULES IMPORTATIONS #####
import utility as u
import matplotlib.pyplot as plt
import plot 
import PCA_LDA as dimRed
import generative_models as gen
##### CLASSES #####


##### FUNCTIONS #####


##### MAIN OF THE PROGRAM #####
if __name__ == "__main__":
    DTR, LTR =u.load(".\dataset\Train.txt")
    DTE, LTE =u.load(".\dataset\Test.txt")

    mu = u.computeMean(DTR)
    C = u.computeCovarianceMatrix(DTR)
    m = 2 #dim 2

    gen.Kfold(DTR,LTR,5)
    
