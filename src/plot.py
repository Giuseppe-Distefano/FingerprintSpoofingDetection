import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utility as u

def plot_scatter_PCA(DP, L):
    DP0 = DP[:, L==0] 
    DP1 = DP[:, L==1]  

    plt.scatter(DP0[0, :], DP0[1, :], label = 'Spoofed')  
    plt.scatter(DP1[0, :], DP1[1, :], label = 'Authentic') 
    
    plt.legend()
    plt.savefig('./images/PCA_2dim' + '.jpg')
    plt.show()        

def plot_hist(DTR,LTR):
    
    for i in range(10):
       labels = ["Spoofed", "Authentic"]
       title ="feature"+ str(i)
       plt.figure()
       plt.title(title)

       y = DTR[:, LTR == 0][i]
       plt.hist(y, bins=40, density=True, alpha=0.4, linewidth=1.0, color='yellow', edgecolor='black',
                label=labels[0])
       y = DTR[:, LTR == 1][i]
       plt.hist(y, bins=40, density=True, alpha=0.4, linewidth=1.0, color='green', edgecolor='black',
                label=labels[1])
       plt.legend()
       plt.savefig('./images/hist_' + title + '.jpg')

def plot_hist_LDA(D,LTR):
       labels = ["Spoofed", "Authentic"]
       plt.figure()

       y = D[:, LTR == 0][0]
       plt.hist(y, bins=60, density=True, alpha=0.5, linewidth=1.0, color='green', edgecolor='black',
                label=labels[0])
       y = D[:, LTR == 1][1]
       plt.hist(y, bins=60, density=True, alpha=0.5, linewidth=1.0, color='red', edgecolor='black',
                label=labels[1])
       plt.legend()
       plt.savefig('./images/hist_LDA'  + '.jpg')
       plt.show()


##### Plot heatmaps #####
def plot_heatmaps (D, L, features, output_folder):
    # All samples
    corr = np.zeros((features, features))
    for x in range(features):
        for y in range(features):
            corr[x][y] = u.compute_correlation(D[x,:], D[y,:])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="YlGnBu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_all.jpeg" % (output_folder))
    
    # Only samples labelled with 0 (spoofed fingerprint samples)
    corr = np.zeros((features, features))
    for x in range(features):
        for y in range(features):
            corr[x][y] = u.compute_correlation(D[x,L==0], D[y,L==0])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="coolwarm", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_spoofed.jpeg" % (output_folder))
    
    # Only samples labelled with 1 (authentic fingerprint samples)
    corr = np.zeros((features, features))
    for x in range(features):
        for y in range(features):
            corr[x][y] = u.compute_correlation(D[x,L==1], D[y,L==1])
    sns.set()
    heatmap = sns.heatmap(np.abs(corr), cmap="BuPu", linewidth=0.3, square=True, cbar=False)
    figure = heatmap.get_figure()
    figure.savefig("%s/heatmap_authentic.jpeg" % (output_folder))
