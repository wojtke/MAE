import argparse
import math

import torch
from tqdm import tqdm
import wandb

from model import MAE_ViT
from utils import setup_seed, visualize
from datasets import load_dataset


def train(args):
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_batch_size = min(args.max_device_batch_size, args.batch_size)
    steps_per_update = args.batch_size // load_batch_size

    train_dataset, val_dataset, train_loader, val_loader = load_dataset(
        dataset=args.dataset,
        batch_size=load_batch_size,
        n_workers=4
    )

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
    loss_fn = lambda img, predicted_img, mask: torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio

    wandb.init(project="mae-test", config=args)
    step_count = 0
    optim.zero_grad()
    progress_bar = tqdm(total=args.total_epoch, initial=0, unit="epoch")
    for epoch in range(1, args.total_epoch+1):
        progress_bar.set_description(f"Pretraining - epoch {epoch}")
        to_log = {}
        model.train()
        train_loss = 0

        for img, _label in train_loader:
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = loss_fn(img, predicted_img, mask)
            loss.backward()
            step_count += 1
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            train_loss += loss.item()
        lr_scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, _label in val_loader:
                img = img.to(device)
                predicted_img, mask = model(img)
                loss = loss_fn(img, predicted_img, mask)
                val_loss += loss.item()

        to_log["train_loss"] = train_loss / len(train_loader)
        to_log["val_loss"] = val_loss / len(val_loader)
        to_log["learning_rate"] = lr_scheduler.get_last_lr()[0]
        to_log["epoch"] = epoch
        to_log["visualisation"] = visualize(
            torch.stack([val_dataset[i][0] for i in range(8)]).to(device)
        )
        wandb.log(to_log, step=step_count)

        msg = f"Epoch {epoch} - train loss: {to_log['train_loss']:.4f}, val loss: {to_log['val_loss']:.4f}"
        progress_bar.set_description(msg)
        progress_bar.update(1)
        print()

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
    parser.add_argument('--dataset', type=str, default='cifar10')
    args = parser.parse_args()

    assert args.batch_size % args.max_device_batch_size == 0

    train(args)
