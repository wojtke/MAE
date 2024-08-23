import argparse
import math

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import wandb
wandb.require("core")

from model import MAE_ViT
from utils import setup_seed, visualize, var_thresholded_l2, compute_l2_per_component, sum_dicts
from datasets import load_dataset
from pca import remove_low_freq


def train(args):
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_batch_size = min(args.max_device_batch_size, args.batch_size)
    steps_per_update = args.batch_size // load_batch_size

    train_dataset, val_dataset, train_loader, val_loader, pca = load_dataset(
        dataset=args.dataset,
        batch_size=load_batch_size,
        n_workers=4
    )
    pca = pca.to(device)
    model = MAE_ViT(mask_ratio=args.mask_ratio, image_size=args.image_size, patch_size=args.patch_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
        
    loss_fn = lambda img_1, img_2, mask: ((img_1 - img_2).square() * mask).mean() / args.mask_ratio

    l2_per_component_list = []

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
            img_high_freq = remove_low_freq(img, pca, var_threshold=args.var_threshold)
            loss = loss_fn(img_high_freq, predicted_img, mask)
            loss.backward()
            step_count += 1
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            train_loss += loss.item()
        lr_scheduler.step()
        to_log["train_loss"] = train_loss / len(train_loader)
        
        model.eval()
        if epoch % args.eval_interval == 0:
            val_loss = 0
            metrics = {}
            l2_per_component = torch.zeros(pca.n_components, device=device)
            l2_per_component_cum = torch.zeros(pca.n_components, device=device)
            l2_per_component_cum_rev = torch.zeros(pca.n_components, device=device)
            with torch.no_grad():
                for img, _label in val_loader:
                    img = img.to(device)
                    img_high_freq = remove_low_freq(img, pca, var_threshold=args.var_threshold)
                    predicted_img, mask = model(img)
                    loss = loss_fn(img_high_freq, predicted_img, mask)
                    val_loss += loss.item()
                    l2_per_component += compute_l2_per_component(
                        img_high_freq, predicted_img, mask, pca, args.mask_ratio, block_size=16)
                    metrics = sum_dicts(
                        metrics,
                        var_thresholded_l2(img, predicted_img, mask, pca, args.mask_ratio, [0.5, 0.6, 0.7, 0.8, 0.9])
                    )
                    
            to_log["val_loss"] = val_loss / len(val_loader)
            l2_per_component_list.append(l2_per_component / len(val_loader))
            to_log.update({k:v/len(val_loader) for k,v in metrics.items()})

            imgs_to_visualize = torch.stack([val_dataset[i][0] for i in range(8)]).to(device)
            vis = visualize(model, imgs_to_visualize, pca, args.var_threshold, denormalize_stats=(0.5, 0.5))
            to_log["visualisation"] = wandb.Image(to_pil_image(vis))
        

        to_log["learning_rate"] = lr_scheduler.get_last_lr()[0]
        to_log["iter"] = step_count

        wandb.log(to_log, step=epoch)

        if epoch % args.snapshot_interval == 0:
            model_filename = f"{args.model_path}.pt"
            torch.save(model.state_dict(), model_filename)
            artifact = wandb.Artifact('model-weights', type='model')
            artifact.add_file(model_filename)
            wandb.log_artifact(artifact)#, aliases=["latest", f"epoch_{epoch}"])

            torch.save(l2_per_component_list, 'l2_per_component_list.pt')
            artifact = wandb.Artifact('l2_per_component', type='metrics')
            artifact.add_file("l2_per_component_list.pt")
            wandb.log_artifact(artifact)#, aliases=["latest", f"epoch_{epoch}"])

        progress_bar.update(1)
        
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model_path', type=str, default='vit-t-mae')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=2)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    
    # Epoch settings
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    
    # Interval settings
    parser.add_argument('--snapshot_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=10)
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--var_threshold', type=float, default=0.8)
    
    args = parser.parse_args()

    assert args.batch_size % args.max_device_batch_size == 0

    train(args)
