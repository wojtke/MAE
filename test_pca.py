import torch
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA, IncrementalPCA as SklearnIncrementalPCA
from sklearn.datasets import load_iris
from torch.testing import assert_close
from pca import PCA, IncrementalPCA

def test_pca_on_iris():
    iris = load_iris()
    data = torch.tensor(iris.data, dtype=torch.float32)
    n_components = 4

    sklearn_pca = SklearnPCA(n_components=n_components)
    sklearn_pca.fit(data.numpy())
    
    torch_pca = PCA(n_components=n_components, n_features=data.shape[1])
    torch_pca.fit(data)

    assert_close(torch.tensor(sklearn_pca.components_), torch_pca.components_, rtol=1e-2, atol=1e-2)
    assert_close(torch.tensor(sklearn_pca.singular_values_), torch_pca.singular_values_, rtol=1e-2, atol=1e-2)
    assert_close(torch.tensor(sklearn_pca.explained_variance_ratio_), torch_pca.explained_variance_ratio_, rtol=1e-2, atol=1e-2)
    print("Test passed: PCA matches Sklearn's PCA on the Iris dataset.")

def test_incremental_pca_on_iris():
    iris = load_iris()
    data = torch.tensor(iris.data, dtype=torch.float32)
    n_components = 4
    batch_size = 50

    sklearn_pca = SklearnIncrementalPCA(n_components=n_components)
    for i in range(0, data.shape[0], batch_size):
        sklearn_pca.partial_fit(data[i:i + batch_size].numpy())
    
    torch_pca = IncrementalPCA(n_components=n_components, n_features=data.shape[1])
    for i in range(0, data.shape[0], batch_size):
        torch_pca.partial_fit(data[i:i + batch_size])
    
    assert_close(torch.tensor(sklearn_pca.components_, dtype=torch.float32), torch_pca.components_, rtol=1e-2, atol=1e-2)
    assert_close(torch.tensor(sklearn_pca.singular_values_, dtype=torch.float32), torch_pca.singular_values_, rtol=1e-2, atol=1e-2)
    assert_close(torch.tensor(sklearn_pca.explained_variance_ratio_, dtype=torch.float32), torch_pca.explained_variance_ratio_, rtol=1e-2, atol=1e-2)
    print("Test passed: IncrementalPCA matches Sklearn's IncrementalPCA on the Iris dataset.")

if __name__ == "__main__":
    test_pca_on_iris()
    test_incremental_pca_on_iris()