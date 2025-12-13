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

def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
     # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
       covmean = covmean.real
     # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid



def log_spectral_distance(x1, x2):
    # Compute the LSD
    difference = torch.log(x1) - torch.log(x2)
    lsd_value = torch.sqrt(torch.mean(difference ** 2, dim=1))
    # Return the average LSD across the three rows (assuming they represent 3 different spectra)
    return lsd_value.mean()



import torch
import torch.nn.functional as F


def calculate_seismic_ssim(target: torch.Tensor, 
                           prediction: torch.Tensor,
                           window_size: int = 51, 
                           sigma: float = 1.5,
                           data_range: float = None,
                           per_sample_range: bool = True) -> torch.Tensor:
    """
    Calculate SSIM for seismic signals of shape [batch_size, 3, 6000]
    
    Args:
        target: Ground truth signal of shape [batch_size, 3, 6000]
        prediction: Predicted signal of shape [batch_size, 3, 6000]
        window_size: Size of the gaussian window (default: 51, larger for time series)
        sigma: Standard deviation of gaussian window (default: 1.5)
        data_range: Range of the data. If None, computed from target
        per_sample_range: If True, compute data range per sample (recommended)
    
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
    
    # Reshape signals to [batch_size, 3, 6000] if needed
    if len(target.shape) == 2:
        target = target.unsqueeze(0)
        prediction = prediction.unsqueeze(0)
    
    batch_size = target.shape[0]
    
    # Compute data range
    if data_range is None:
        if per_sample_range:
            # Compute per-sample range and average
            target_flat = target.reshape(batch_size, -1)
            data_range = (target_flat.max(dim=1)[0] - target_flat.min(dim=1)[0]).mean()
        else:
            # Global range across all data
            data_range = target.max() - target.min()
        
        # Ensure data_range is not too small
        data_range = torch.clamp(data_range, min=1e-8)
    
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create gaussian window
    window = _create_gaussian_window(window_size, sigma, target.device)
    
    # Use reflection padding (better for non-periodic signals)
    pad = window_size // 2
    target_pad = F.pad(target, (pad, pad), mode='reflect')
    prediction_pad = F.pad(prediction, (pad, pad), mode='reflect')
    
    # Ensure same dtype
    if target_pad.dtype != window.dtype:
        window = window.to(target_pad.dtype)
    
    # Calculate means using grouped convolution
    mu_t = F.conv1d(target_pad, window, groups=3)
    mu_p = F.conv1d(prediction_pad, window, groups=3)
    mu_t_sq = mu_t ** 2
    mu_p_sq = mu_p ** 2
    mu_tp = mu_t * mu_p
    
    # Calculate variances and covariance
    sigma_t_sq = F.conv1d(target_pad ** 2, window, groups=3) - mu_t_sq
    sigma_p_sq = F.conv1d(prediction_pad ** 2, window, groups=3) - mu_p_sq
    sigma_tp = F.conv1d(target_pad * prediction_pad, window, groups=3) - mu_tp
    
    # Ensure non-negative variances (numerical stability)
    sigma_t_sq = torch.clamp(sigma_t_sq, min=0)
    sigma_p_sq = torch.clamp(sigma_p_sq, min=0)
    
    # Calculate SSIM
    numerator = (2 * mu_tp + C1) * (2 * sigma_tp + C2)
    denominator = (mu_t_sq + mu_p_sq + C1) * (sigma_t_sq + sigma_p_sq + C2)
    ssim_map = numerator / denominator
    
    # Average over all dimensions
    return ssim_map.mean()


# Alternative: Per-channel SSIM (returns SSIM for each seismic component)
def calculate_seismic_ssim_per_channel(target: torch.Tensor, prediction: torch.Tensor,
                                       window_size: int = 51, sigma: float = 1.5,
                                       data_range: float = None) -> dict:
    """
    Calculate SSIM separately for each seismic channel (Z, N, E)
    
    Returns:
        dict: {'mean': overall_ssim, 'channel_0': ssim_z, 'channel_1': ssim_n, 'channel_2': ssim_e}
    """
    
    def _create_gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        gauss = torch.exp(
            -torch.arange(-(window_size // 2), window_size // 2 + 1, device=device) ** 2 
            / (2 * sigma ** 2)
        )
        return (gauss / gauss.sum()).view(1, 1, -1)
    
    if target.shape != prediction.shape:
        raise ValueError(f"Shapes don't match: {target.shape} vs {prediction.shape}")
    
    if len(target.shape) == 2:
        target = target.unsqueeze(0)
        prediction = prediction.unsqueeze(0)
    
    batch_size, num_channels, _ = target.shape
    window = _create_gaussian_window(window_size, sigma, target.device).to(target.dtype)
    pad = window_size // 2
    
    ssim_channels = {}
    
    for ch in range(num_channels):
        # Extract single channel
        target_ch = target[:, ch:ch+1, :]
        pred_ch = prediction[:, ch:ch+1, :]
        
        # Compute data range for this channel
        if data_range is None:
            ch_range = target_ch.max() - target_ch.min()
            ch_range = torch.clamp(ch_range, min=1e-8)
        else:
            ch_range = data_range
        
        C1 = (0.01 * ch_range) ** 2
        C2 = (0.03 * ch_range) ** 2
        
        # Padding and convolution
        target_pad = F.pad(target_ch, (pad, pad), mode='reflect')
        pred_pad = F.pad(pred_ch, (pad, pad), mode='reflect')
        
        mu_t = F.conv1d(target_pad, window)
        mu_p = F.conv1d(pred_pad, window)
        mu_t_sq = mu_t ** 2
        mu_p_sq = mu_p ** 2
        mu_tp = mu_t * mu_p
        
        sigma_t_sq = F.conv1d(target_pad ** 2, window) - mu_t_sq
        sigma_p_sq = F.conv1d(pred_pad ** 2, window) - mu_p_sq
        sigma_tp = F.conv1d(target_pad * pred_pad, window) - mu_tp
        
        sigma_t_sq = torch.clamp(sigma_t_sq, min=0)
        sigma_p_sq = torch.clamp(sigma_p_sq, min=0)
        
        numerator = (2 * mu_tp + C1) * (2 * sigma_tp + C2)
        denominator = (mu_t_sq + mu_p_sq + C1) * (sigma_t_sq + sigma_p_sq + C2)
        ssim_ch = (numerator / denominator).mean()
        
        ssim_channels[f'channel_{ch}'] = ssim_ch.item()
    
    ssim_channels['mean'] = sum(ssim_channels.values()) / num_channels
    
    return ssim_channels




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



def highfrequency_metrics(y_pred,y,):
    fft_ypred = torch.fft.rfft(y_pred)
    fft_y = torch.fft.rfft(y)
    fft_ypred = torch.abs(fft_ypred)
    fft_y = torch.abs(fft_y)

if __name__ == "__main__":
    # Test highfrequency_metrics
    y = torch.randn(8,3,6000)
    y_pred = y + torch.randn_like(y) * 0.05
    ssim = calculate_seismic_ssim(y_pred,y)
    print(f"SSIM calculated successfully. : {ssim}")