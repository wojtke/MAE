
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize

def load_dataset(batch_size, n_workers=4):
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=n_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=n_workers)

    return train_dataset, val_dataset, train_loader, val_loader
    