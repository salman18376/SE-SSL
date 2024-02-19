
import os
import torch
import torchaudio
import numpy as np
import argparse
import librosa
import scipy
from tqdm import tqdm

PCS = [
    1.0, 1.0, 1.0, 1.0702, 1.0702, 1.1825,
    1.1825, 1.1825, 1.2877, 1.2877, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.4, 1.4, 1.3228, 1.3228, 1.3228, 1.3228,
    1.3228, 1.3228, 1.3228, 1.3228, 1.3228, 1.3228,
    1.3228, 1.3228, 1.3228, 1.3228, 1.3228, 1.3228,
    1.3228, 1.3228, 1.3228, 1.3228, 1.3228, 1.3228,
    1.3228, 1.2386, 1.2386, 1.2386, 1.2386, 1.2386,
    1.2386, 1.2386, 1.2386, 1.2386, 1.2386, 1.2386,
    1.2386, 1.2386, 1.2386, 1.2386, 1.2386, 1.2386,
    1.2386, 1.2386, 1.2386, 1.2386, 1.2386, 1.2386,
    1.2386, 1.2386, 1.2386, 1.2386, 1.1614, 1.1614,
    1.1614, 1.1614, 1.1614, 1.1614, 1.1614, 1.1614,
    1.1614, 1.1614, 1.1614, 1.1614, 1.1614, 1.1614,
    1.1614, 1.1614, 1.1614, 1.1614, 1.1614, 1.1614,
    1.1614, 1.1614, 1.1614, 1.1614, 1.1614, 1.1614,
    1.1614, 1.1614, 1.1614, 1.1614, 1.1614, 1.1614,
    1.1614, 1.0772, 1.0772, 1.0772, 1.0772, 1.0772,
    1.0772, 1.0772, 1.0772,
]

print("PCS length: ", len(PCS))

PCS = np.array(PCS)
N_FFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
Fs = 16000
higher_bin = 201

maxv = np.iinfo(np.int16).max


def Sp_and_phase(signal):
    signal_length = signal.shape[0]
    n_fft = N_FFT
    y_pad = librosa.util.fix_length(signal, size = signal_length + n_fft // 2)

    F = librosa.stft(y_pad, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH, window=scipy.signal.hamming)

    Lp = PCS * np.transpose(np.log1p(np.abs(F)), (1, 0))
    phase = np.angle(F)

    NLp = np.transpose(Lp, (1, 0))

    return NLp, phase, signal_length


def SP_to_wav(mag, phase, signal_length):
    mag = np.expm1(mag)
    Rec = np.multiply(mag, np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=HOP_LENGTH,
                           win_length=WINDOW_LENGTH,
                           window=scipy.signal.hamming, length=signal_length)
    return result


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in tqdm(files):
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


parser = argparse.ArgumentParser()

parser.add_argument('--input_folder', default="/home/salman/SE_Self-Supervise_Learning-/reconstructed_audio/mag_only/1_log1p_compression_l1_waveform_consestisy_loss", type=str)
parser.add_argument('--output_folder', default="/home/salman/SE_Self-Supervise_Learning-/reconstructed_audio/mag_only/1_log1p_compression_l1_waveform_consestisy_loss1", type=str)

args = parser.parse_args()


# ---------- validation data ---------- #
Test_Noisy_paths = get_filepaths(args.input_folder)
Output_path = args.output_folder

if Output_path[-1] != '/':
    Output_path = Output_path + '/'

for i in tqdm(Test_Noisy_paths):
    noisy_wav, _ = torchaudio.load(i)
    noisy_LP, Nphase, signal_length = Sp_and_phase(noisy_wav.squeeze().numpy())

    enhanced_wav = SP_to_wav(noisy_LP, Nphase, signal_length)
    enhanced_wav = enhanced_wav/np.max(abs(enhanced_wav))

    torchaudio.save(
        Output_path+i.split('/')[-1],
        torch.unsqueeze(torch.from_numpy(enhanced_wav).type(torch.float32), 0),
        Fs
    )
