################################
#     MODULES IMPORTATIONS     #
################################
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import modules.utility as utility


############################
#     GLOBAL VARIABLES     #
############################
features = int(10)
distinct_classes = int(2)


#####################
#     FUNCTIONS     #
#####################
# ----- Read file -----
def read_file (filename):
    D = []
    L = []
    with open(filename) as file:
        for line in file:
            try:
                attributes = line.split(",")[0:features]
                attributes = utility.row_to_column(np.array([float(i) for i in attributes]))
                label = int(line.split(",")[-1].strip())
                D.append(attributes)
                L.append(label)
            except:
                pass
    return np.hstack(D), np.array(L, dtype=np.int32)


# ----- Load training set -----
def load_training_set (training_filename):
    DTR,LTR = read_file(training_filename)
    return DTR,LTR


# ----- Load test set -----
def load_test_set (test_filename):
    DTE,LTE = read_file(test_filename)
    return DTE,LTE