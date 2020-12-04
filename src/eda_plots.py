import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import offsetbox 
import matplotlib.colors as mcolors

import collections
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import manifold
import umap

holdoutseed = 4

def pca_sample_data(data, features):
    """
    Returns feature and target datasets with a 4 to 1 ratio of 
    non-fraud to fraud samples of the data. 
    """
    minority_class = data[data['Class'] == 1]
    SampleCount = 4 * len(minority_class)
    majority_class = data[data['Class'] == 0].sample(SampleCount)
    data_sampled = majority_class.append([minority_class])
    X_sampled = data_sampled[features]
    y_sampled = data_sampled['Class']
    return X_sampled, y_sampled
    

def embedding_plot(ax, X ,labels,title):
    """
    Returns a plot of data X
    """
    
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=mcolors.ListedColormap(["lightgrey", "black"]), s=8) 
    ax.gca().set_facecolor((1, 1, 1))
    ax.set_xlabel('1st dimension')
    ax.set_ylabel('2nd dimension')
    ax.grid(False)
    ax.set_title(title)
    ax.legend()

def pca_reduction(X_sampled):
    X_tsne = manifold.TSNE(n_components=2, 
                       init='pca',
                       perplexity=30,
                       learning_rate=200,
                       n_iter=500,
                       random_state=2).fit_transform(X_sampled)
    X_umap = umap.UMAP(n_neighbors=5, 
                   min_dist=0.4, 
                   n_components=2, 
                   random_state=2).fit_transform(X_sampled)
    return X_tsne, X_umap

def plot_cumm_pca(ax, X_sampled, components):

    pca = decomposition.PCA(n_components=components)
    pca.fit_transform(X_sampled)

    x = range(0, X_sampled.shape[1])
    y = np.cumsum(pca.explained_variance_ratio_)
    
    ax.scatter(x, y)
    ax.set_title("PCA Components vs. Cummulative Explained Variance")
    ax.set_xlabel('Components')
    ax.set_ylabel('Variance Explained')

if __name__ == '__main__':

    data = pd.read_csv("../data/creditcard.csv")
    features = ['V%d' % number for number in range(1, 29)]

    X_sampled, y_sampled = pca_sample_data(data, features)

    fig, ax = plt.subplots()
    plot_cumm_pca(ax, X_sampled, 29)
    print(f'Cummulative Total Explained Variance of first \
            10 components: \
            {np.sum(pca.explained_variance_ratio_[0:10]):.2f}')

    fig, axs = plt.subplots(1, 2, figsize=(14,6))
    embedding_plot(axs[0], X_tsne, y_sampled, \
        "t-SNE 2D plot of Fraud vs. Non Fraud using PCA dimensionality reduction")
    fig.savefig("../images/pca_visualization.png")
#    embedding_plot(axs[1], X_umap, y_sampled,"umap")