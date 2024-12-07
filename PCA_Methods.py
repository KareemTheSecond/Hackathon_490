import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans


def PCA_PlotClusters(df5, NumberOfClusters): 
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df5)
    
    # Apply Gaussian Mixture Model (GMM) for clustering
    GMM = GaussianMixture(n_components=NumberOfClusters, random_state=42)
    GMM.fit(X_pca)
    
    ClustersPred = GMM.predict(X_pca) # WILL RETURN! 
    MeansMatrix = GMM.means_
    CovarianceMatrix = GMM.covariances_
    
    # Visualization
    Colors = ["b", "g", "r", "c", "m", "y", "k", "navy", "coral", "slategray"]
    plt.figure(figsize=(10, 8))
    ClusterColors = [Colors[label % len(Colors)] for label in ClustersPred]
    
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=ClusterColors, alpha=0.6)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("PCA and GMM Clustering: Selected FEATURES")
    
    ax = plt.gca()
    for i, (mean, covariance) in enumerate(zip(MeansMatrix, CovarianceMatrix)):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(mean, width=width, height=height, angle=angle, 
                          color=Colors[i % len(Colors)], alpha=0.3)
        ax.add_patch(ellipse)
    
    plt.show()
    return ClustersPred
    
def ElbowBestCluster(df5):
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(df5)
  wcss = []
    
  for n in range(1, 10):
      kmeans = KMeans(n_clusters=n, random_state=42)
      kmeans.fit(X_Pca)
      wcss.append(kmeans.inertia_)
    
    # Plot the Elbow Curve
  plt.figure(figsize=(8, 4))
  plt.plot(range(1, 10), wcss, marker='o')
  plt.xlabel('Number of Clusters')
  plt.ylabel('WCSS')
  plt.title('Elbow Method for Optimal Clusters')
  plt.show()
    
def BicsBestCluster(df5): 
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df5)
    n_components_range = range(1, 11)  # Try 1 to 10 clusters
    bics = []

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X_pca)
        bics.append(gmm.bic(X_pca))
  
    best_n = n_components_range[np.argmin(bics)]
    
    plt.figure(figsize=(8,5))
    plt.plot(n_components_range, bics, marker='o')
    plt.xlabel('Number of Clusters (Components)')
    plt.ylabel('BIC Score')
    plt.title('BIC Scores for Different Numbers of Clusters')
    plt.legend()
    plt.show()

      
