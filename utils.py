import torch
import torch.nn as nn
import torchmetrics
import os
import argparse
import json

from data.audio_processor import AudioProcessor
from data.audio_dataset import AudioDataset
from data.compressor import SpectrogramCompressor



def get_pred_waveforms(predicted_magnitude, predicted_phase, compressor, audio_processor, input_waveforms, norm_factors=None):
    '''
    Gets the predicted waveforms from the predicted magnitude and phase.
    :param predicted_magnitude: A batch of predicted magnitudes.
    :param predicted_phase: A batch of predicted phases.
    :param compressor: The compressor to use.
    :param audio_processor: The audio processor to use.
    :param input_waveforms: The input waveforms to use for the target length.
    :return: The predicted waveforms.
    '''
    predicted_waveforms = []
    for i in range(predicted_magnitude.shape[0]):
        if compressor is not None: m_pred = compressor.decompress(predicted_magnitude[i])
        else: m_pred = predicted_magnitude[i]
        p_pred = predicted_phase[i]
        w = audio_processor.get_waveform_from_spectrogram(
            mag=m_pred,
            phase=p_pred,
            target_length=input_waveforms.shape[1]
        )
        predicted_waveforms.append(w)
    predicted_waveforms = torch.stack(predicted_waveforms)
    if norm_factors is not None:
        for i in range(predicted_waveforms.shape[0]):
            predicted_waveforms[i] = predicted_waveforms[i] * norm_factors[i]
    return predicted_waveforms



def get_target_waveforms(targ_waveforms, target_mag, target_phase, audio_processor, compressor=None):
    target_waveforms = []

    for i in range(target_mag.shape[0]):
        # Decompress the target magnitude if compression is enabled
        if compressor is not None:
            mag_target = compressor.decompress(target_mag[i])
        else:
            mag_target = target_mag[i]

        # Convert the target magnitude and output phase to a waveform
        target_waveform = audio_processor.get_waveform_from_spectrogram(mag_target,
                                                                      target_phase[i],
                                                                      target_length=targ_waveforms.shape[1])
        target_waveforms.append(target_waveform)

    # Convert the predicted waveforms to a spectrogram
    target_waveforms = torch.stack(target_waveforms, dim=0)
    
    return target_waveforms

