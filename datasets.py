import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from pca import PCA

def load_dataset(dataset, batch_size, n_workers=4):
    if dataset == "cifar10":
        return load_cifar10(batch_size, n_workers)
    elif dataset == "stl10":
        return load_stl10_unsupervised(batch_size, n_workers)
    else:
        raise NotImplementedError(f"No such dataset.")

def load_cifar10(batch_size, n_workers, pca_path="pca_cifar10.pt"):
    normalize_stats = (0.5, 0.5)
    transform = Compose([ToTensor(), Normalize(*normalize_stats)])

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=n_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=n_workers, drop_last=True)

    pca = PCA(n_components=1024, n_features=3*32*32)
    pca.load_state_dict(torch.load(pca_path, weights_only=True))
    pca.add_normalization(*normalize_stats)

    return train_dataset, val_dataset, train_loader, val_loader, pca


def load_stl10_unsupervised(batch_size, n_workers, pca_path="pca_stl10.pt"):
    normalize_stats = (0.5, 0.5)
    transform = Compose([ToTensor(), Normalize(*normalize_stats)])

    train_dataset = torchvision.datasets.STL10('data', split="train+unlabeled", download=True, transform=transform)
    val_dataset = torchvision.datasets.STL10('data', split="test", download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=n_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=n_workers, drop_last=True)

    pca = PCA(n_components=1024, n_features=3*96*96)
    pca.load_state_dict(torch.load(pca_path, weights_only=True))
    pca.add_normalization(*normalize_stats)

    return train_dataset, val_dataset, train_loader, val_loader, pca
