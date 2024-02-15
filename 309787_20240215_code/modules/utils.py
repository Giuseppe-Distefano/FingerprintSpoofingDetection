##### LIBRARIES #####
import numpy


##### FUNCTIONS #####
# --- Represent an array as a row vector ---
def vrow (v):
    v = v.reshape(1, v.size)
    return v
    

# --- Represent an array as a column vector ---
def vcol (v):
    v = v.reshape(v.size, 1)
    return v
    

# --- ___ ---
def randomize (DTR, LTR, seed):
    numpy.random.seed(seed) 
    dtr,ltr = numpy.zeros((10, DTR.shape[1])), numpy.zeros((LTR.size,))
    indexes = numpy.random.permutation(DTR.shape[1])
    i = 0
    for ind in indexes:
        dtr[:,i] = DTR[:,ind]
        ltr[i] = LTR[ind]
        i+=1
        
    return dtr, ltr


# --- Compute mean ---
def compute_mean (X):
    return vcol(X.mean(1))


# --- Compute covariance matrix ---
def compute_covariance (X):
    mu = vcol(X.mean(1))
    centered= X - mu
    C = numpy.dot(centered, centered.transpose()) / X.shape[1]
    return C


# --- Compute ZNorm ---
def compute_znorm (DTR, DTE):
    mu_r = vcol(DTR.mean(1))
    std_r = vcol(DTR.std(1))

    DTR_z = (DTR - mu_r) / std_r
    DTE_z = (DTE - mu_r) / std_r
    return DTR_z, DTE_z


# --- Load data from files ---
def load_data (filename):
    data, labels = [], []
    
    with open(filename) as f:
        for line in f:
            v = line.split(",")[0:10]
            attr = vcol(numpy.array(v, dtype=numpy.float64))
            data.append(attr)
            labels.append(line.split(",")[10])
            
    D = numpy.hstack(data)  
    L = numpy.asarray(labels, dtype=numpy.int32)
    return D,L


# --- Principal Components Analysis ---
def apply_PCA (X, m):
    mu = vcol(X.mean(1))
    centered = X - mu
    C = numpy.dot(centered, centered.transpose()) / X.shape[1]
    _,U = numpy.linalg.eigh(C)
    P = U[:,::-1][:,0:m]
    return P
   

# --- Predict labels ---
def predict_labels (llr, t): 
    predicted = numpy.zeros(len(llr))
    for i in  range(len(llr)):
        if(llr[i] > t): predicted[i] = 1
        else: predicted[i] = 0
    return predicted


# --- Count mispredictions ---
def count_mispredictions (predicted_labels, LTE) :
    err = LTE - predicted_labels
    wp = 0
    for i in range(err.shape[0]):
        if(err[i]!=0): wp += 1
    return wp


# --- ___ ---
def proportional_labels (LTR0, LTR1):
    new_array = []
    num_ones = 0
    zeros_b4_ones = 2
    
    for i in range(len(LTR0)):
        if num_ones == zeros_b4_ones:
            new_array.append(LTR1[0])
            LTR1 = LTR1[1:]
            num_ones = 0
        
            if len(LTR1) == 0: break
            if i%50 == 0:
                new_array.append(LTR1[0])
                LTR1 = LTR1[1:]

        new_array.append(LTR0[i])
        num_ones += 1
        if len(LTR1) == 0: break
    if len(LTR1) > 0: new_array.extend(LTR1)
    
    return new_array


# --- ___ ---
def proportional_dataset (DTR0, DTR1):
    num_ones = 0
    zeros_b4_ones = 2
    
    new_matrix = numpy.empty((10, 0))
    num_ones = 0
    for i in range(DTR0.shape[1]):
        if num_ones == zeros_b4_ones:
            new_matrix = numpy.concatenate([new_matrix, DTR1[:,0:1]], axis=1)
            DTR1 = DTR1[:,1:]
            num_ones = 0
            
            if DTR1.shape[1] == 0: break
            if i%50 == 0:
                new_matrix = numpy.concatenate([new_matrix, DTR1[:, 0:1]], axis=1)
                DTR1 = DTR1[:, 1:]
        
        new_matrix = numpy.concatenate([new_matrix, DTR0[:, i:i+1]], axis=1)
        num_ones += 1
        if DTR1.shape[1] == 0: break
    if DTR1.shape[1] > 0: new_matrix = numpy.concatenate([new_matrix, DTR1], axis=1)
    return new_matrix


# --- Compute correlation ---
def compute_correlation (X, Y):
    x_sum, y_sum = numpy.sum(X), numpy.sum(Y)
    x2_sum, y2_sum = numpy.sum(X**2), numpy.sum(Y**2)

    sum_cp = numpy.sum(X * Y.T)
    n = X.shape[0]
    num = n*sum_cp - x_sum*y_sum
    den = numpy.sqrt((n*x2_sum - x_sum**2) * (n*y2_sum - y_sum**2))

    return num/den
