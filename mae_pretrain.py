import os
import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
import wandb
from model import *
from utils import setup_seed

@torch.no_grad()
def visualize(model, val_img):
    ''' visualize the first 16 predicted images on val dataset'''
    predicted_val_img, mask = model(val_img)
    predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
    img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
    img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
    return img


def train(args):
    setup_seed(args.seed)
    load_batch_size = min(args.max_device_batch_size, args.batch_size)
    steps_per_update = args.batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_loader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, load_batch_size, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()

    progress_bar = tqdm(range(args.total_epoch), desc="Epoch 0", position=0)
    for e in range(args.total_epoch):
        to_log = {}
        model.train()
        train_loss = 0
        for img, _label in train_loader:
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            train_loss += loss.item()
        lr_scheduler.step()
        to_log["train_loss"] = train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            for img, _label in val_loader:
                img = img.to(device)
                predicted_img, mask = model(img)
                val_loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            val_loss += loss.item()
        lr_scheduler.step()
        to_log["val_loss"] = val_loss / len(val_loader)

        to_log["visualisation"] = visualize(
            torch.stack([val_dataset[i][0] for i in range(8)] + [train_dataset[i][0] for i in range(8)]).to(device)
        )

        wandb.log(to_log, step=step_count)

    torch.save(model, args.model_path)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    args = parser.parse_args()

    assert args.batch_size % args.load_batch_size == 0

    train(args)

