
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize

def load_dataset(dataset, batch_size, n_workers=4):
    if dataset == "cifar10":
        return load_cifar10(batch_size, n_workers)
    elif dataset == "stl10":
        return load_stl10_unsupervised(batch_size, n_workers)
    else:
        raise NotImplementedError(f"No such dataset.")

def load_cifar10(batch_size, n_workers):
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=n_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=n_workers)

    return train_dataset, val_dataset, train_loader, val_loader


def load_stl10_unsupervised(batch_size, n_workers):
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])

    train_dataset = torchvision.datasets.STL10('data', split="train+unlabeled", download=True, transform=transform)
    val_dataset = torchvision.datasets.STL10('data', split="test", download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=n_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=n_workers)

    return train_dataset, val_dataset, train_loader, val_loader
