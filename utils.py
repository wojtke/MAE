import random
import torch
import numpy as np
from einops import rearrange
import wandb
import torch
import torch.nn.functional as F

from pca import remove_low_freq

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# @torch.no_grad()
# def visualize(model, val_img):
#     predicted_val_img, mask = model(val_img)
#     predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
#     img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
#     img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
#     img = img.permute(1, 2, 0).cpu().numpy()
#     img = np.clip(img * 255, 0, 255).astype(np.uint8)
#     wandb_image = wandb.Image(img)
#     return wandb_image

@torch.no_grad()
def visualize(model, img, pca, var_threshold, denormalize_stats=None):
    """Visualizes the original, masked, predicted, and high-frequency images in a grid."""
    pred_img, mask = model(img)
    masked_img = img * (1-mask)
    img_high_freq = remove_low_freq(img, pca, var_threshold=var_threshold)
    pred_img = pred_img * mask + img_high_freq * (1 - mask)
    img = torch.cat([img, masked_img, pred_img, img_high_freq], dim=0)
    img = rearrange(img, '(n b) c h w -> c (n h) (b w)', n=4)
    if denormalize_stats is not None:
        mean, std = denormalize_stats
        img = img * std + mean
    img = img.clip(0, 1)
    return img
    # img = img.permute(1, 2, 0).cpu().numpy()
    # img = np.clip(img * 255, 0, 255).astype(np.uint8)
    # wandb_image = wandb.Image(img)
    # return wandb_image

def var_thresholded_metrics(l2_per_component_cum, pca, thresholds):
    cumulative_var = torch.cumsum(pca.explained_variance_ratio_, dim=0)
    total_l2 = l2_per_component_cum[-1]

    metrics = {}
    for th in thresholds:
        idx = (cumulative_var <= th).sum().item() + 1
        metrics[f"val_l2/bottom_{1-th:.2f}"] = total_l2-l2_per_component_cum[idx].item()

    return metrics
        

def calc_var_thresholded_metrics(img, predicted_img, mask, mask_ratio, pca):
    img = img.view(img.size(0), -1)
    predicted_img = predicted_img.view(predicted_img.size(0), -1)
    mask = mask.view(mask.size(0), -1)

    l1 = lambda img_1, img_2: torch.mean((img_1 - img_2).abs() * mask) / mask_ratio
    l2 = lambda img_1, img_2: torch.mean((img_1 - img_2).square() * mask) / mask_ratio
    
    cumulative_var = torch.cumsum(pca.explained_variance_ratio_, dim=0)
    top_50_idx = (cumulative_var <= 0.5).sum().item() + 1
    top_75_idx = (cumulative_var <= 0.75).sum().item() + 1
    top_90_idx = (cumulative_var <= 0.9).sum().item() + 1

    results = {}
    results['l2_total'] = l2(img, predicted_img)
    results['l1_total'] = l1(img, predicted_img)

    img_transformed = pca.transform(img)
    predicted_img_transformed = pca.transform(predicted_img)
    
    img_top_50 = pca.inverse_transform(img_transformed, n=top_50_idx)
    predicted_img_top_50 = pca.inverse_transform(predicted_img_transformed, n=top_50_idx)
    results['l2_top_50'] = l2(img_top_50, predicted_img_top_50)
    results['l1_top_50'] = l1(img_top_50, predicted_img_top_50)
    
    img_top_75 = pca.inverse_transform(img_transformed, n=top_75_idx)
    predicted_img_top_75 = pca.inverse_transform(predicted_img_transformed, n=top_75_idx)
    results['l2_top_75'] = l2(img_top_75, predicted_img_top_75)
    results['l1_top_75'] = l1(img_top_75, predicted_img_top_75)

    img_top_90 = pca.inverse_transform(img_transformed, n=top_90_idx)
    predicted_img_top_90 = pca.inverse_transform(predicted_img_transformed, n=top_90_idx)
    results['l2_top_90'] = l2(img_top_75, predicted_img_top_90)
    results['l1_top_90'] = l1(img_top_75, predicted_img_top_90)
    
    img_bot_50 = pca.inverse_transform(img_transformed, n=top_50_idx, top=False)
    predicted_img_bot_50 = pca.inverse_transform(predicted_img_transformed, n=top_50_idx, top=False)
    results['l2_bottom_50'] = l2(img_bot_50, predicted_img_bot_50)
    results['l1_bottom_50'] = l1(img_bot_50, predicted_img_bot_50)
    
    img_bot_25 = pca.inverse_transform(img_transformed, n=top_75_idx, top=False)
    predicted_img_bot_25 = pca.inverse_transform(predicted_img_transformed, n=top_75_idx, top=False)
    results['l2_bottom_25'] = l2(img_bot_25, predicted_img_bot_25)
    results['l1_bottom_25'] = l1(img_bot_25, predicted_img_bot_25)

    img_bot_10 = pca.inverse_transform(img_transformed, n=top_90_idx, top=False)
    predicted_img_bot_10 = pca.inverse_transform(predicted_img_transformed, n=top_90_idx, top=False)
    results['l2_bottom_10'] = l2(img_bot_10, predicted_img_bot_10)
    results['l1_bottom_10'] = l2(img_bot_10, predicted_img_bot_10)

    results = {k:v.item() for k,v in results.items()}
    
    return results

@torch.no_grad()
def compute_l2_per_component(img, pred_img, mask, pca, mask_ratio, block_size=64):
    """
    Computes the L2 loss per PCA component between the original and predicted images.

    Args:
        img (torch.Tensor): Original image [batch_size, channels, height, width].
        pred_img (torch.Tensor): Predicted image, same shape as `img`.
        mask (torch.Tensor): Mask for weighting loss, same shape as `img`.
        pca (nn.Module): Trained PCA module with components and mean.
        block_size (int, optional): Block size for memory efficiency. Default is 64.

    Returns:
        torch.Tensor: L2 loss per PCA component [num_components].
        torch.Tensor: Cumulative L2 loss per component.
    """
    img = img.view(img.size(0), -1)
    pred_img = pred_img.view(pred_img.size(0), -1)
    mask = mask.view(mask.size(0), -1)

    img_t = (img - pca.mean_) @ pca.components_.T
    pred_img_t = (pred_img - pca.mean_) @ pca.components_.T
    
    result = torch.zeros(pca.n_components, dtype=torch.float32, device=img.device)
    result_cum = torch.zeros(pca.n_components, dtype=torch.float32, device=img.device)
    num_samples = img_t.size(0)
    
    for start_idx in range(0, num_samples, block_size):
        end_idx = min(start_idx + block_size, num_samples)
        img_t_block = img_t[start_idx:end_idx]
        pred_img_t_block = pred_img_t[start_idx:end_idx]
        diff_block = img_t_block - pred_img_t_block
        weighted_diff_block = diff_block.unsqueeze(2) * pca.components_.unsqueeze(0) * mask[start_idx:end_idx].unsqueeze(1)
        result += torch.mean(weighted_diff_block.square(), dim=(0, 2)) / mask_ratio
        weighted_diff_block = torch.cumsum(weighted_diff_block, dim=1)
        result_cum += torch.mean(weighted_diff_block.square(), dim=(0, 2)) / mask_ratio

    result /= (num_samples / block_size)
    result_cum /= (num_samples / block_size)
    return result, result_cum
    

def sum_dicts(*dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
    return result


