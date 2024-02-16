import numpy as np
import torch
import torch as nn
import torch.nn.functional as F
import torch.nn as nn

from torch_pesq import PesqLoss
import auraloss


from data.audio_dataset import AudioProcessor




############################################
# waveform_loss_wsdr
############################################
    
def weighted_sdr_loss(noisy_waveform, predicted_waveform, clean_wavefrom, eps=1e-8):
     
    y_pred = predicted_waveform
    y_true = clean_wavefrom
    x = noisy_waveform

    def sdr_fn(true, pred, eps=1e-8):
        # Calculate the Signal-to-Distortion Ratio (SDR)
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return - (num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred
    
    # Calculate the weighting factor 'a' to balance the contribution of speech and noise
    # It is calculated as the ratio of the power of the clean speech to the sum of the power of clean speech and noise
    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    
    # Calculate weighted Signal-to-Distortion Ratio (wSDR)
    # It combines the SDR of the speech component and the SDR of the noise component
    # The speech component is weighted by 'a', and the noise component is weighted by (1 - a)
    # The weighted SDR is the final evaluation metric for the enhanced waveform
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

############################################
# Phase loss with anti-wrapping
############################################

def anti_wrapping_function(x):
    """
    Applies an anti-wrapping function to a tensor x. The anti-wrapping function
    "wraps" the values in x to the range [0, 2 * np.pi) by dividing x by 2 * np.pi,
    rounding the result to the nearest integer, and multiplying the result by 2 * np.pi.
    This function returns the absolute value of the wrapped tensor.
    """
    return  torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi) #torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def phase_loss(
        predicted,
        target
    ):
        '''
        Computes different losses based on Group Delay (GD) and Instantaneous Amplitude Fluctuations (IAF).
        :param predicted: torch.Tensor. Batch of predicted phases.
        :param target: torch.Tensor. Batch of target phases.
        :return: torch.Tensor. loss computed.
        '''

        
        f_dim = predicted.shape[-1]
        n_frames = predicted.shape[-2]
        
        GD_matrix = torch.triu(torch.ones(f_dim, f_dim), diagonal=1) - torch.triu(torch.ones(f_dim, f_dim), diagonal=2) - torch.eye(f_dim)
        GD_matrix = GD_matrix.to(predicted.device)

        IAF_matrix = torch.triu(torch.ones(n_frames, n_frames), diagonal=1) - torch.triu(torch.ones(n_frames, n_frames), diagonal=2) - torch.eye(n_frames)
        IAF_matrix = IAF_matrix.to(predicted.device)
   
        
        # Calculate the group delay of reference_phase and predicted_phase:
        GD_reference = torch.matmul(target, GD_matrix)
        GD_predicted = torch.matmul(predicted, GD_matrix)
       
        # Calculate the instantaneous amplitude fluctuations of reference_phase and predicted_phase:
        IAF_reference = torch.matmul(target.permute(0,2,1), IAF_matrix)
        IAF_predicted = torch.matmul(predicted.permute(0,2,1), IAF_matrix)

        # Calculate the three loss values:
        L_IP = torch.mean(anti_wrapping_function(target-predicted))
        L_GD = torch.mean(anti_wrapping_function(GD_reference-GD_predicted))
        L_IAF = torch.mean(anti_wrapping_function(IAF_reference-IAF_predicted))
        

        # Return the three loss values as a tuple:
        l_overall = L_IP + L_GD + L_IAF
        return l_overall




def waveform_mrstft(
    predicted,
    target,
):
    '''
    Computes the MultiResolutionSTFT loss (from auraloss) between the predicted and target waveforms.
    :param predicted: torch.Tensor. Batch of predicted waveforms.
    :param target: torch.Tensor. Batch of target waveforms.
    :param kwargs: dict. Dictionary with extra arguments needed to compute the loss.
    :return: torch.Tensor. loss computed.
    '''

    # if self.use_consistent_waveform:
    #     target = self.get_consistent_waveform(target, **kwargs)

    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 512],#{512, 1024, 2048},
    hop_sizes=[120, 240, 50],# 50, 120, 240}
    win_lengths=[600, 1200, 240], # {240, 600, 1200}.
    scale="mel",
    n_bins=128,
    sample_rate=16000,
    perceptual_weighting=True,
)

    if len(predicted.shape) <= 2: # add channel dimension
        predicted = predicted.unsqueeze(dim=1)
    if len(target.shape) <= 2: # add channel dimension
        target = target.unsqueeze(dim=1)

    return mrstft(predicted, target)

