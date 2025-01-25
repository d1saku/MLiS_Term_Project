import pandas as pd
import numpy as np
import scipy as sp

class PCA_my:
    def __init__(self, n_components):
        self.n_components_ = n_components
        self.mean_ = None
        self.std_ = None
        self.pcs_ = None
        self.explained_variance_ = None
        self.loadings = None


    def fit(self, X):
        self.mean_ = X.mean(axis=0)

        cov_matrix = np.cov(X - self.mean_, rowvar=False)

        eigenvalues, eigenvectors = self._eigen_decomp(cov_matrix)

        self.pcs_ = eigenvectors[:, :self.n_components_]
        self.explained_variance_ = eigenvalues
        self.loadings_ = eigenvectors * np.sqrt(eigenvalues)

        return self


    def transform(self, X):
        X = X - self.mean_
        return np.dot(X, self.pcs_)


    def _eigen_decomp(self, cov_matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        return eigenvalues, eigenvectors

pca = PCA_my( n_components=2)

np.random.seed(42)
data = np.random.rand(10, 2)
print("Data:\n", data)

data_transformed = pca.fit(data).transform(data)


print("my pca: \n", data_transformed)
print("Loadings: \n", pca.loadings_)



# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# pca.fit(data)

# x_pca = pca.transform(data)

# print("sklearn_pca: \n", x_pca)