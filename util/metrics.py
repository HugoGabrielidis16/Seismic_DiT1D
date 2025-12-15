import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from obspy.signal.tf_misfit import eg, pg
import torch.nn.functional as F


def snr(pred, target, eps=1e-8):
    noise = target - pred
    signal_power = torch.mean(target**2, dim=-1)
    noise_power = torch.mean(noise**2, dim=-1)
    return 10 * torch.log10((signal_power + eps) / (noise_power + eps)).mean()

def MSE(pred, target):
    loss = nn.MSELoss()
    loss_v = loss(pred, target)
    return loss_v

def gof(pred,target):
    eg_value = eg(pred,target)
    pg_value = pg(pred,target)
    return eg_value, pg_value

def compute_embeddings(autoencoder_model,dataloader, count):
    image_embeddings = []


    for _ in tqdm(range(count)):
        images = next(iter(dataloader))
        embeddings = autoencoder_model.predict(images)
        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)



def log_spectral_distance(x1, x2):
    difference = torch.log(x1) - torch.log(x2)
    lsd_value = torch.sqrt(torch.mean(difference ** 2, dim=1))

    return lsd_value.mean()



def calculate_seismic_ssim(target: torch.Tensor, 
                           prediction: torch.Tensor,
                           window_size: int = 51, 
                           sigma: float = 1.5,
                           data_range: float = None,
                           per_sample_range: bool = True) -> torch.Tensor:
    """
    Calculate the 1d SSIM  between target and prediction
    """
    
    def _create_gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        gauss = torch.exp(
            -torch.arange(-(window_size // 2), window_size // 2 + 1, device=device) ** 2 
            / (2 * sigma ** 2)
        )
        window = gauss / gauss.sum()
        return window.view(1, 1, -1).repeat(3, 1, 1)
    
    if target.shape != prediction.shape:
        raise ValueError(f"Shapes don't match: {target.shape} vs {prediction.shape}")
    
    if len(target.shape) == 2:
        target = target.unsqueeze(0)
        prediction = prediction.unsqueeze(0)
    
    batch_size = target.shape[0]
    
    if data_range is None:
        if per_sample_range:
            target_flat = target.reshape(batch_size, -1)
            data_range = (target_flat.max(dim=1)[0] - target_flat.min(dim=1)[0]).mean()
        else:
            data_range = target.max() - target.min()
        
        data_range = torch.clamp(data_range, min=1e-8)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    window = _create_gaussian_window(window_size, sigma, target.device)
    
    pad = window_size // 2
    target_pad = F.pad(target, (pad, pad), mode='reflect')
    prediction_pad = F.pad(prediction, (pad, pad), mode='reflect')
    
    if target_pad.dtype != window.dtype:
        window = window.to(target_pad.dtype)
    
    mu_t = F.conv1d(target_pad, window, groups=3)
    mu_p = F.conv1d(prediction_pad, window, groups=3)
    mu_t_sq = mu_t ** 2
    mu_p_sq = mu_p ** 2
    mu_tp = mu_t * mu_p
    
    sigma_t_sq = F.conv1d(target_pad ** 2, window, groups=3) - mu_t_sq
    sigma_p_sq = F.conv1d(prediction_pad ** 2, window, groups=3) - mu_p_sq
    sigma_tp = F.conv1d(target_pad * prediction_pad, window, groups=3) - mu_tp
    
    sigma_t_sq = torch.clamp(sigma_t_sq, min=0)
    sigma_p_sq = torch.clamp(sigma_p_sq, min=0)
    
    numerator = (2 * mu_tp + C1) * (2 * sigma_tp + C2)
    denominator = (mu_t_sq + mu_p_sq + C1) * (sigma_t_sq + sigma_p_sq + C2)
    ssim_map = numerator / denominator
    return ssim_map.mean()

def spectral_similarity(y_pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute spectral similarity between predicted and target signals.
    
    Args:
        y_pred: Predicted signal tensor of shape [Bs, 3, 6000]
        y: Target signal tensor of shape [Bs, 3, 6000]
        eps: Small value to avoid division by zero
        
    Returns:
        torch.Tensor: Spectral similarity score (lower is better)
    """
    # Compute FFT for both signals
    fft_pred = torch.fft.fft(y_pred, dim=-1)
    fft_target = torch.fft.fft(y, dim=-1)
    
    # Compute magnitude spectra (absolute values)
    mag_pred = torch.abs(fft_pred)
    mag_target = torch.abs(fft_target)
    
    # Normalize the spectra
    mag_pred_norm = mag_pred / (torch.sum(mag_pred, dim=-1, keepdim=True) + eps)
    mag_target_norm = mag_target / (torch.sum(mag_target, dim=-1, keepdim=True) + eps)
    
    # Compute L1 distance between normalized spectra
    spectral_distance = torch.mean(torch.abs(mag_pred_norm - mag_target_norm))
    
    return spectral_distance


def spectral_similarity_with_freq(y_pred: torch.Tensor, y: torch.Tensor, 
                                sampling_rate: float = 100.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute spectral similarity with explicit frequency calculation.
    
    Args:
        y_pred: Predicted signal tensor of shape [Bs, 3, 6000]
        y: Target signal tensor of shape [Bs, 3, 6000]
        sampling_rate: Sampling rate in Hz
        eps: Small value to avoid division by zero
    """
    # Compute FFT
    fft_pred = torch.fft.fft(y_pred, dim=-1)
    fft_target = torch.fft.fft(y, dim=-1)
    
    # Calculate frequency axis (similar to np.fft.fftfreq)
    n_samples = y_pred.shape[-1]
    freqs = torch.fft.fftfreq(n_samples, d=1/sampling_rate)
    
    # Get positive frequencies mask
    pos_freq_mask = freqs > 0
    
    # Compute magnitude spectra for positive frequencies
    mag_pred = torch.abs(fft_pred[..., pos_freq_mask])
    mag_target = torch.abs(fft_target[..., pos_freq_mask])
    
    # Normalize the spectra
    mag_pred_norm = mag_pred / (torch.sum(mag_pred, dim=-1, keepdim=True) + eps)
    mag_target_norm = mag_target / (torch.sum(mag_target, dim=-1, keepdim=True) + eps)
    spectral_distance = torch.mean(torch.abs(
        mag_pred_norm- mag_target_norm
    ))
    return spectral_distance


