import argparse
import math

import torch
from tqdm import tqdm
import wandb
wandb.require("core")

from model import MAE_ViT, ViT_Classifier
from utils import setup_seed
from datasets import load_dataset

def train(args):
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_batch_size = min(args.max_device_batch_size, args.batch_size)
    steps_per_update = args.batch_size // load_batch_size

    train_dataset, val_dataset, train_loader, val_loader, _ = load_dataset(
        dataset=args.dataset,
        batch_size=load_batch_size,
        n_workers=4
    )
    model = MAE_ViT(image_size=args.image_size, patch_size=args.patch_size)
    
    if args.pretrained_model_path is not None:
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location='cpu'))
    model = ViT_Classifier(model.encoder, num_classes=10).to(device)

    if not args.linear_probe:
        optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
    else:
        optim = torch.optim.SGD(model.head.parameters(), lr=args.base_learning_rate * args.batch_size / 256, momentum=0.9)
        lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.total_epoch, eta_min=0)
   
    loss_fn = torch.nn.CrossEntropyLoss()

    wandb.init(project="mae-test", config=args)
    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    progress_bar = tqdm(total=args.total_epoch, initial=0, unit="epoch")
    for epoch in range(1, args.total_epoch+1):
        progress_bar.set_description(f"Training - epoch {epoch}")
        to_log = {}
        
        model.train()
        train_loss, correct = 0, 0
        for img, label in train_loader:
            img = img.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            loss.backward()
            step_count += 1
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            train_loss += loss.item()
            correct += (logits.argmax(dim=-1) == label).sum().item()
        lr_scheduler.step()
        to_log["train_loss"] = train_loss / len(train_loader)
        to_log["train_acc"] = correct / len(train_dataset)
        
        model.eval()
        if epoch % args.eval_interval == 0:
            val_loss, correct = 0, 0
            with torch.no_grad():
                for img, label in val_loader:
                    img = img.to(device)
                    logits = model(img)
                    loss = loss_fn(logits, label)
                    val_loss += loss.item()
                    correct += (logits.argmax(dim=-1) == label).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = correct / len(val_dataset)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_filename = f"{args.output_model_path}_best.pt"
                torch.save(model.state_dict(), best_model_filename)
                artifact = wandb.Artifact('model-weights-best-acc', type='model')
                artifact.add_file(best_model_filename)
                wandb.log_artifact(artifact)
                print(f"Best model saved with accuracy: {best_val_acc:.4f}")

        
            to_log["val_loss"] = val_loss
            to_log["val_acc"] = val_acc

        to_log["learning_rate"] = lr_scheduler.get_last_lr()[0]
        to_log["iter"] = step_count

        wandb.log(to_log, step=epoch)

        if epoch % args.snapshot_interval == 0 or epoch == args.total_epoch:
            model_filename = f"{args.output_model_path}.pt"
            torch.save(model.state_dict(), model_filename)
            artifact = wandb.Artifact(f'model-weights-{epoch}', type='model')
            artifact.add_file(model_filename)
            wandb.log_artifact(artifact)

        progress_bar.update(1)
        
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=2)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    # Epoch settings
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    
    # Interval settings
    parser.add_argument('--snapshot_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=10)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--linear_probe', action="store_true")


    args = parser.parse_args()

    assert args.batch_size % args.max_device_batch_size == 0

    assert not (args.linear_probe and args.pretrained_model_path is None)

    if args.output_model_path is None:
        model = args.pretrained_model_path.replace(".pt", "") if args.pretrained_model_path else "vit-t"
        mode = "linear_probe" if args.linear_probe else ("finetune" if args.pretrained_model_path else "from_scratch"
        args.output_model_path =  f"{model}_{mode}.pt"

    train(args)
