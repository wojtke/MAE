import random
import torch
import numpy as np
from einops import rearrange
import torch

from pca import remove_low_freq

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def visualize(model, img, pca, var_threshold, denormalize_stats=None) -> torch.Tensor:
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


@torch.no_grad()
def var_thresholded_metrics(l2pc_cum, l2pc_cum_rev, pca, thresholds) -> dict:
    """
    Computes L2 norm metrics based on PCA explained variance thresholds.

    Args:
        l2pc_cum (torch.Tensor): Cumulative L2 norms of PCA components, 
            starting from the top variance components (highest explained variance).
        l2pc_cum_rev (torch.Tensor): Cumulative L2 norms of PCA components, 
            starting from the bottom variance components (lowest explained variance).
        pca (PCA): Fitted PCA object with `explained_variance_ratio_`.
        thresholds (list of float): List of variance thresholds (0-1). Counting from the top variance.

    Returns:
        dict: Metrics as L2 norms for top and bottom components at each threshold.
    """
    cumulative_var = torch.cumsum(pca.explained_variance_ratio_, dim=0)

    metrics = {}
    for th in thresholds:
        idx = (cumulative_var <= th).sum().item() + 1
        metrics[f"val_l2/top_{th:.2f}"] = l2pc_cum[idx].item()
        metrics[f"val_l2/bottom_{1-th:.2f}"] = l2pc_cum_rev[idx].item()

    return metrics
        

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
    result_cum_reverse = torch.zeros(pca.n_components, dtype=torch.float32, device=img.device)
    num_samples = img_t.size(0)
    
    for start_idx in range(0, num_samples, block_size):
        end_idx = min(start_idx + block_size, num_samples)
        img_t_block = img_t[start_idx:end_idx]
        pred_img_t_block = pred_img_t[start_idx:end_idx]
        diff_block = img_t_block - pred_img_t_block
        weighted_diff_block = diff_block.unsqueeze(2) * pca.components_.unsqueeze(0) * mask[start_idx:end_idx].unsqueeze(1)
        
        result += weighted_diff_block.square().mean(dim=(0, 2)) / mask_ratio
        weighted_diff_block_cum = torch.cumsum(weighted_diff_block, dim=1)
        weighted_diff_block_cum_reverse = weighted_diff_block + weighted_diff_block_cum[:, -1:] - weighted_diff_block_cum
        result_cum += weighted_diff_block_cum.square().mean(dim=(0, 2)) / mask_ratio
        result_cum_reverse += weighted_diff_block_cum_reverse.square().mean(dim=(0, 2)) / mask_ratio

    result *= block_size / (num_samples * mask_ratio)
    result_cum *= block_size / (num_samples * mask_ratio)
    result_cum_reverse *= block_size / (num_samples * mask_ratio)
    return result, result_cum, result_cum_reverse
    

def sum_dicts(*dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
    return result


