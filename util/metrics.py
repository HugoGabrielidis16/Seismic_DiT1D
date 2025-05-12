import numpy as np
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from obspy.signal.tf_misfit import eg, pg,plot_tf_gofs
from numpy import linalg
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




def log_spectral_distance(x1, x2):
    # Compute the LSD
    difference = torch.log(x1) - torch.log(x2)
    lsd_value = torch.sqrt(torch.mean(difference ** 2, dim=1))
    # Return the average LSD across the three rows (assuming they represent 3 different spectra)
    return lsd_value.mean()





def calculate_seismic_ssim(target: torch.Tensor, prediction: torch.Tensor,
                        window_size: int = 11, sigma: float = 1.5,
                        data_range: float = None) -> torch.Tensor:
    """
    Calculate SSIM for seismic signals of shape [batch_size, 3, 6000]
    
    Args:
        target: Ground truth signal of shape [batch_size, 3, 6000]
        prediction: Predicted signal of shape [batch_size, 3, 6000]
        window_size: Size of the gaussian window (default: 11)
        sigma: Standard deviation of gaussian window (default: 1.5)
        data_range: Range of the data. If None, uses max-min of target
    
    Returns:
        torch.Tensor: Single SSIM value averaged over batch and channels
    """
    
    def _create_gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """
        Create a gaussian window for SSIM calculation
        """
        gauss = torch.exp(
            -torch.arange(-(window_size // 2), window_size // 2 + 1, device=device) ** 2 
            / (2 * sigma ** 2)
        )
        window = gauss / gauss.sum()
        # Reshape for grouped conv1d [3, 1, window_size]
        return window.view(1, 1, -1).repeat(3, 1, 1)
    
    if target.shape != prediction.shape:
        raise ValueError(f"Shapes don't match: {target.shape} vs {prediction.shape}")
    
    if data_range is None:
        data_range = target.max() - target.min()
    
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create gaussian window
    window = _create_gaussian_window(window_size, sigma, target.device)
    
    # Reshape signals to [batch_size, 3, 6000]
    if len(target.shape) == 2:
        target = target.unsqueeze(0)
        prediction = prediction.unsqueeze(0)
    
    # Handle padding - using circular padding as alternative to reflect
    pad = window_size // 2
    target_pad = torch.cat([target[..., -pad:], target, target[..., :pad]], dim=-1)
    prediction_pad = torch.cat([prediction[..., -pad:], prediction, prediction[..., :pad]], dim=-1)
    
    # Calculate means
    if target_pad.dtype != window.dtype:
        window = window.to(target.dtype)
    if prediction_pad.dtype != window.dtype:
        prediction_pad = prediction_pad.to(window.dtype)

    mu_t = F.conv1d(target_pad, window, groups=3)
    mu_p = F.conv1d(prediction_pad, window, groups=3)
    mu_t_sq = mu_t ** 2
    mu_p_sq = mu_p ** 2
    mu_tp = mu_t * mu_p
    
    # Calculate variances and covariance
    sigma_t_sq = F.conv1d(target_pad * target_pad, window, groups=3) - mu_t_sq
    sigma_p_sq = F.conv1d(prediction_pad * prediction_pad, window, groups=3) - mu_p_sq
    sigma_tp = F.conv1d(target_pad * prediction_pad, window, groups=3) - mu_tp
    
    # Calculate SSIM
    num = (2 * mu_tp + C1) * (2 * sigma_tp + C2)
    den = (mu_t_sq + mu_p_sq + C1) * (sigma_t_sq + sigma_p_sq + C2)
    ssim_map = num / den
    
    # Average over the signal length, channels, and batch
    return ssim_map.mean()

def highfrequency_metrics(y_pred,y,):
    fft_ypred = torch.fft.rfft(y_pred)
    fft_y = torch.fft.rfft(y)
    fft_ypred = torch.abs(fft_ypred)
    fft_y = torch.abs(fft_y)


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
    #mag_pred_norm = mag_pred / (torch.norm(mag_pred, p=2) + eps)
    #mag_target_norm = mag_target / (torch.sum(mag_target,p=2) + eps)
    
    # Compute distance only for frequencies below 30Hz
    #freq_mask_30hz = freqs[pos_freq_mask] <= 30
    #spectral_distance = torch.mean(torch.abs(
    #    mag_pred_norm[..., freq_mask_30hz] - mag_target_norm[..., freq_mask_30hz]
    #))
    spectral_distance = torch.mean(torch.abs(
        mag_pred_norm- mag_target_norm
    ))
    
    return spectral_distance


if __name__ == "__main__":
    # Test with batch dimension
    x = torch.randn(2, 3, 6000)
    y = torch.randn(2, 3, 6000)
    print(spectral_similarity(x,y))