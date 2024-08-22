# adapted from https://github.com/gngdb/pytorch-pca

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import math


class PCA(nn.Module):
    def __init__(self, n_components: int, n_features: int):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.register_buffer("mean_", torch.zeros((1, n_features), dtype=torch.float32))
        self.register_buffer("components_", torch.zeros((n_components, n_features), dtype=torch.float32))
        self.register_buffer("singular_values_", torch.zeros(n_components, dtype=torch.float32))
        self.normalize = False


    def _validate_data(self, X):
        assert X.shape[0] >= self.n_components
        assert X.shape[1] >= self.n_components
        assert X.device == self.mean_.device

    @property
    def explained_variance_ratio_(self) -> torch.Tensor:
        explained_variance = (self.singular_values_ ** 2) / (self.n_features - 1)
        total_variance = explained_variance.sum()
        return explained_variance / total_variance

    def _svd(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)
        return U, S, Vt

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for
        deterministic output.

        This method ensures that the output remains consistent across different
        runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular
              vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular
              vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        n_examples, n_features = X.size()
        assert n_features == self.n_features
        assert n_examples >= self.n_components
        
        self.mean_ = X.mean(0, keepdim=True)
        U, S, Vt = self._svd(X - self.mean_)
        
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]

        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            X = X * self.normalize_std + self.normalize_mean

        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y: torch.Tensor, n:int=None, top: bool=True) -> torch.Tensor:
        """
        Transform vector back from principal component space.

        Args:
            Y (torch.Tensor): Vector in PC space.
            n (int, optional): Use reduced number of components for the transform
            top (bool, optional): If using reduced number of components for the transform,
                    whether to use top n components (high variance) (top=True, default)
                    or bottom n components (low variance) (top=False).

        Returns:
            (torch.Tensor): Vectors after inverse transform.
        """        
        if n:
            components = self.components_[:n] if top else self.components_[n:]
            Y = Y[:, :n] if top else Y[:, n:]
        else:
            components = self.components_
        X = Y @ components + self.mean_
        if self.normalize:
            X = (X - self.normalize_mean) / self.normalize_std
        return X

    def add_normalization(self, mean, std):
        """Adjusts PCA parameters to match the normalized input."""
        self.normalize = True
        self.normalize_mean, self.normalize_std = (mean, std)


class IncrementalPCA(PCA):
    def __init__(self, n_components: int, n_features: int):
        super().__init__(n_components, n_features)
        self.n_samples_seen_ = 0
        
    def partial_fit(self, X):
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py
        if self.n_samples_seen_ == 0:
            self.n_samples_seen_ += X.shape[0]
            return self.fit(X)
        
        batch_size = X.shape[0]
        batch_mean = X.mean(dim=0, keepdim=True)
        total_samples = self.n_samples_seen_ + batch_size
        new_mean = (self.n_samples_seen_ * self.mean_ + batch_size * batch_mean) / total_samples
        
        X -= batch_mean
        mean_correction = math.sqrt(
            (self.n_samples_seen_ / total_samples) * batch_size
        ) * (self.mean_ - batch_mean)
        X = torch.vstack(
                (torch.diag(self.singular_values_) @ self.components_,
                 X,
                 mean_correction)
            )

        U, S, Vt = self._svd(X)
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]
        self.mean_ = new_mean
        self.n_samples_seen_ = total_samples

        return self
        


def remove_high_freq(img: torch.Tensor, pca: PCA, var_threshold: float = 0.5) -> torch.Tensor:
    """
    Remove high frequency (low variance) components from an image using PCA.

    Args:
        img (torch.Tensor): A 4D tensor representing the image.
        pca (PCA): Pre-fitted PCA object for the dataset the image comes from.
        var_threshold (float): Cumulative variance (0 to 1) to remove. 

    Returns:
        torch.Tensor: Image with high frequency components removed, in the original shape.
    """
    b, c, h, w = img.size()
    img = img.view(b, c*h*w)
    img_transformed = pca.transform(img)
    cumulative_var = torch.cumsum(pca.explained_variance_ratio_, dim=0)
    idx = (cumulative_var <= 1-var_threshold).sum().item() + 1
    img = pca.inverse_transform(img_transformed, n=idx)
    img = img.view(b, c, h, w)
    return img

def remove_low_freq(img, pca, var_threshold=0.5):
    """
    Remove low frequency (high variance) components from an image using PCA.

    Args:
        img (torch.Tensor): A 4D tensor representing the image.
        pca (PCA): Pre-fitted PCA object for the dataset the image comes from.
        var_threshold (float): Cumulative variance (0 to 1) to remove. 

    Returns:
        torch.Tensor: Image with low frequency components removed, in the original shape.
    """
    b, c, h, w = img.size()
    img = img.view(b, c*h*w)
    img_transformed = pca.transform(img)
    cumulative_var = torch.cumsum(pca.explained_variance_ratio_, dim=0)
    idx = (cumulative_var <= var_threshold).sum().item() + 1
    img = pca.inverse_transform(img_transformed, n=idx, top=False)
    img = img.view(b, c, h, w)
    return img   


if __name__ == "__main__":
    from tqdm import tqdm
    import torch
    from torchvision.datasets import CIFAR10, STL10
    from torchvision.transforms import ToTensor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 4096
    transform = ToTensor()
    dataset = CIFAR10('data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=4)
    
    pca = IncrementalPCA(n_components=1024, n_features=3*32*32).to(device)
    for img, _label in tqdm(dataloader, desc="Doing PCA on CIFAR10"):
        img = img.view(img.size(0), -1).to(device)
        pca.partial_fit(img)
    
    torch.save(pca.cpu().state_dict(), "pca_cifar.pt")
    

    batch_size = 2048
    transform = ToTensor()
    dataset = STL10('data', split="train+unlabeled", download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=4)
    
    pca = IncrementalPCA(n_components=1024, n_features=3*96*96).to(device)
    for img, _label in tqdm(dataloader, desc="Doing PCA on STL10"):
        img = img.view(img.size(0), -1).to(device)
        pca.partial_fit(img)
    
    torch.save(pca.cpu().state_dict(), "pca_stl10.pt")
