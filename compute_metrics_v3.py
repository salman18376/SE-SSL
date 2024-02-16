import argparse
import os
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoModel

from data.compressor import SpectrogramCompressor
from data.audio_dataset import AudioDataset
from data.audio_processor import AudioProcessor
from model.speech_enhamcement_v3 import SpeechEnhancementModel
import json

from scipy.io import wavfile
import librosa
from tqdm import tqdm

from pesq import pesq
from pystoi import stoi

import matplotlib.pyplot as plt
import numpy as np



import torchmetrics
import scipy

# Create the argument parser
parser = argparse.ArgumentParser()
# ------------------------------------------------ Experiment configuration ------------------------------------------------

parser.add_argument("--experiment_config", type=str, help="path to experiment configuration file", required=True)

# ------------------------------------------------ Setup parameters ------------------------------------------------

parser.add_argument("--model_checkpoint", type=str, help="path to the saved model checkpoint to be used for inference", required=True)
parser.add_argument("--cuda", action="store_true", help="use cuda")
parser.add_argument("--reconstructed_audio_folder", type=str, help="path to the folder where the reconstructed audio files will be saved", required=True)
parser.add_argument("--model_tag", type=str, help="model tag for the SSL model", required=True)
parser.add_argument("--magnitude_head", type=str, default=None, help="head to use for the magnitude spectrogram")
parser.add_argument("--phase_head", type=str, default=None, help="head to use for the phase spectrogram")
parser.add_argument("--complex_head", type=str, default=None, help="head to use for the complex spectrogram")


parser.add_argument("--freeze_ssl", action="store_true", help="freeze the weights of the SSL model")
parser.add_argument("--phase_input", type=str, choices=["predicted", "reference", "input"], default="predicted",
                    help="Phase input for waveform reconstruction")
parser.add_argument("--magnitude_input", type=str, default="predicted", choices=["predicted", "reference", "input"], help="magnitude input for waveform reconstruction")
parser.add_argument("--enable_pcs_on_output", action="store_true", help="enable PCS on the output magnitude spectrogram")
parser.add_argument("--skip_generation", action="store_true", help="skip the generation of the enhanced audio files")

# ------------------------------------------------ Debugging parameters ------------------------------------------------

parser.add_argument("--generate_images", action="store_true", help="generate images having the original and enhanced waveforms overlapped")

args = parser.parse_args()


# ------------------------------------------------
# 
#              Setting the parameters 
#
# ------------------------------------------------

with open(args.experiment_config) as f:
    config = json.load(f)

if torch.cuda.is_available() and args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ------------------------------------------------
# 
#              Setting the parameters 
#
# ------------------------------------------------

# --------------- Backbone model ------------------

ssl_model = AutoModel.from_pretrained(args.model_tag, output_hidden_states=config["use_all_layers"], output_attentions=config["use_all_layers"])
if config["duplicate_from_cnn"]:
    ssl_model.feature_extractor.conv_layers[6].conv.stride = (1,)
# ssl_model = ssl_model.encoder
# if config["duplicate_from_cnn"]:
    
    # ssl_model.conv2.stride = (1,)
    
# # --------------- Compressor ------------------

if config["use_compression"]:
    compressor = SpectrogramCompressor(
        alpha=config["compressor_alpha"], 
        learnable=config["learnable_compression"], 
        compressor_type=config["compression_type"]
    )

    if config["learnable_compression"]:
        filename = args.model_checkpoint.split("/")[-1]
        base_path = args.model_checkpoint.replace(filename, "")
        compressor_filename = f"compressor_{filename}"
        compressor.load_state_dict(torch.load(os.path.join(base_path, compressor_filename), map_location=device))

else: compressor = None




# --------------- Speech Enhancement Model ------------------

model = SpeechEnhancementModel(
    ssl_model=ssl_model,
    freeze_ssl=args.freeze_ssl,
    magnitude_head=args.magnitude_head,
    phase_head=args.phase_head,
    complex_head=args.complex_head,
    ssl_embedding_dim=ssl_model.config.hidden_size,
    stft_embedding_dim=config["stft_embedding_dim"],
    type=config["type"],
    sigmoid_type=config["sigmoid_type"],
    compressor=compressor,
    use_all_layers=config["use_all_layers"],
).to(device)

# Load the saved model parameters
model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
model.eval()

audio_processor = AudioProcessor(
    sample_rate=config["sample_rate"],
    max_length_in_seconds=config["max_length_in_seconds"],
    n_fft=config["n_fft"],
    hop_length=config["hop_length"],
    win_length=config["win_length"],
    normalized_stft = config["normalized_stft"],
)


# --------------- Dataset ---------------
# Create the dataset for the noisy audio files
test_dataset = AudioDataset(
    noisy_path=config["noisy_test_folder"],
    clean_path=config["clean_test_folder"],
    n_fft=config["n_fft"],
    hop_length=config["hop_length"],
    win_length=config["win_length"],
    ssl_model_tag=args.model_tag,
    max_length_in_seconds = config["max_length_in_seconds"],
    randomly_sample_n_samples= config["randomly_sample_n_samples_testset"],
    normalize_waveform = config["normalize_waveform"]

)

# create output folder if it does not exist
if not os.path.exists(args.reconstructed_audio_folder):
    os.makedirs(args.reconstructed_audio_folder)


PCS_VECTORS = {
    400: [
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
}


def apply_pcs_to_waveform(waveform, n_fft, hop_length=160, window_length=400):

    if torch.is_tensor(waveform):
        if waveform.device != "cpu":
            waveform = waveform.cpu()
        waveform = waveform.numpy()
    
    pcs_vec = np.array(PCS_VECTORS[n_fft])
    signal_length = waveform.shape[0]
    # apply padding to the waveform
    y_pad = librosa.util.fix_length(waveform, size = signal_length + n_fft // 2)
    # compute the STFT
    stft_output = librosa.stft(
        y_pad, n_fft=n_fft, hop_length=hop_length, 
        win_length=window_length, window=scipy.signal.hamming
    )
    
    l_p = pcs_vec * np.transpose(np.log1p(np.abs(stft_output)), (1, 0))
    phase = np.angle(stft_output)
    nl_p = np.transpose(l_p, (1, 0))

    # return NLp, phase, signal_length

    mag = np.expm1(nl_p)
    rec = np.multiply(mag, np.exp(1j*phase))
    result = librosa.istft(
        rec, hop_length=hop_length, win_length=window_length,
        window=scipy.signal.hamming, length=signal_length
    )

    result = result/np.max(abs(result))
    
    if not torch.is_tensor(waveform):
        result = torch.tensor(result, dtype=torch.float32)
        
    
    return result

# ------------------------------------------------
# 
#            Infer enhanced waveforms
#
# ------------------------------------------------

if not args.skip_generation:
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), total=len(test_dataset), desc="Generating enhanced audio files"):
            filename = test_dataset.filenames[i]
            original_audio, original_sr = torchaudio.load(config["noisy_test_folder"] + filename)
            norm_factor = torch.sqrt(original_audio.shape[1] / torch.sum(original_audio ** 2))
            clean_audio , original_sr_c = torchaudio.load(config["clean_test_folder"] + filename)
            len_original_audio = original_audio.shape[1]

            batch = test_dataset.__getitem__(i)
            for k, v in batch.items():
                batch[k] = v.unsqueeze(dim=0).to(device)

            input_mag = batch["input_stft"]
            input_phase = batch["input_phase"]

            # ----------------- Compress the input spectrogram -----------------
            if compressor is not None:
                input_mag = compressor.compress(input_mag)
                batch["input_stft"] = input_mag

            predicted_mag, predicted_phase  = model(batch)

            if config["type"] == "masking" and args.magnitude_head is not None:
                predicted_mag = predicted_mag * input_mag
            
            

            predicted_mag = predicted_mag.squeeze(0)
            predicted_phase = predicted_phase.squeeze(0)
            
            
            # ----------------- Select the magnitude_input for waveform reconstruction -----------------
            if args.magnitude_input == "reference":
                selected_mag = batch["output_stft"]
            elif args.magnitude_input == "input":
                selected_mag = batch["input_stft"]
            else: 
                args.magnitude_input == "predicted"
                selected_mag = predicted_mag
                
                
            # ----------------- Select the phase input for waveform reconstruction -----------------
            if args.phase_input == "reference":
                selected_phase = batch["output_phase"]
            elif args.phase_input == "input":
                selected_phase = batch["input_phase"]
            else: 
                args.phase_input == "predicted"
                selected_phase = predicted_phase

            # ----------------- Decompress the predicted spectrogram -----------------
            if compressor is not None: mag_predicted = compressor.decompress(predicted_mag)
            else: mag_predicted = predicted_mag
            
            
            # ----------------- Select the magnitude_input for waveform reconstruction -----------------
            if args.magnitude_input == "reference": selected_mag = batch["output_stft"].squeeze(0) # use the reference magnitude
            elif args.magnitude_input == "input": selected_mag = batch["input_stft"].squeeze(0) # use the noisy magnitude 
            elif args.magnitude_input == "predicted": selected_mag = mag_predicted # use the predicted magnitude
            else: raise NotImplementedError(f"magnitude_input {args.magnitude_input} not implemented")
                
                
            # ----------------- Select the phase input for waveform reconstruction -----------------
            if args.phase_input == "reference": selected_phase = batch["output_phase"].squeeze(0) # use the reference phase
            elif args.phase_input == "input": selected_phase = batch["input_phase"].squeeze(0) # use the noisy phase
            elif args.phase_input == "predicted": selected_phase = predicted_phase # use the predicted phase
            else: raise NotImplementedError(f"phase_input {args.phase_input} not implemented")

            enhanced_wav = audio_processor.get_waveform_from_spectrogram(selected_mag, selected_phase,target_length=len_original_audio)
            

            enhanced_wav = enhanced_wav / norm_factor
            if args.enable_pcs_on_output:
                enhanced_wav = apply_pcs_to_waveform(enhanced_wav, config["n_fft"], config["hop_length"], config["win_length"])
            
            enhanced_wav = enhanced_wav.unsqueeze(dim=0)
            
            # Save the reconstructed waveform to a WAV file with the same name as the original file
            output_filename = os.path.splitext(os.path.basename(filename))[0] + ".wav"
            output_file = os.path.join(args.reconstructed_audio_folder, output_filename)
            torchaudio.save(output_file, enhanced_wav.to("cpu"), config["sample_rate"])

            if args.generate_images:
                # plot the original and enhanced waveforms
                plt.figure(figsize=(20, 5))
                # plt.plot(original_audio[0].numpy(), label='noisy')
                plt.plot(enhanced_wav[0].to("cpu").numpy(), label='enhanced', alpha=0.7)
                plt.plot(clean_audio[0].numpy(), label='clean')
                plt.legend()
                plt.savefig(os.path.join(args.reconstructed_audio_folder, os.path.splitext(os.path.basename(filename))[0] + ".png"))
                plt.close()


# ------------------------------------------------
# 
#            Infer enhanced waveforms
#
# ------------------------------------------------

def compute_metrics(clean_dir, enhanced_dir):
    pesq_metric = torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality(config["sample_rate"], 'wb')
    stoi_metric = torchmetrics.audio.stoi.ShortTimeObjectiveIntelligibility(config["sample_rate"], False)
    pesq_scores = []
    stoi_scores = []
        
    def get_scores(clean_dir, enhanced_dir, filename):
        clean_path = os.path.join(clean_dir, filename)
        enhanced_path = os.path.join(enhanced_dir, filename)
        clean_waveform, _ = torchaudio.load(clean_path)
        enhanced_waveform, _ = torchaudio.load(enhanced_path)

        # check if the lengths are the same - in case trim
        if clean_waveform.shape[1] != enhanced_waveform.shape[1]:
            print(f"Lengths of clean and enhanced waveforms are different for {filename}. Trimming...")
            print(f"Clean waveform length: {clean_waveform.shape[1]}")
            print(f"Enhanced waveform length: {enhanced_waveform.shape[1]}")
            min_length = min(clean_waveform.shape[1], enhanced_waveform.shape[1])
            clean_waveform = clean_waveform[:, :min_length]
            enhanced_waveform = enhanced_waveform[:, :min_length]
        try:
            pesq_score = pesq_metric(enhanced_waveform, clean_waveform)
            stoi_score = stoi_metric(enhanced_waveform, clean_waveform)
        except Exception as e: 
            print(e)
            print(filename)
            pesq_score = 0
            stoi_score = 0
        
        return pesq_score, stoi_score
    
    
    # parallelize the computation of the metrics
    from joblib import Parallel, delayed
    # tqdm is used to show the progress bar
    scores = Parallel(n_jobs=16)(delayed(get_scores)(clean_dir, enhanced_dir, filename) for filename in tqdm(os.listdir(clean_dir), desc="Computing metrics"))
    pesq_scores = [score[0] for score in scores]
    stoi_scores = [score[1] for score in scores]

    avg_pesq = sum(pesq_scores) / len(pesq_scores)
    avg_stoi = sum(stoi_scores) / len(stoi_scores)
    
    return avg_pesq, avg_stoi

clean_dir = config["clean_test_folder"]
enhanced_dir = args.reconstructed_audio_folder
avg_pesq_score, avg_stoi_score = compute_metrics(clean_dir, enhanced_dir)
print(f"*" * 50)
print(f"Results for {args.model_checkpoint}")
print(f"*" * 50)


print (f"Magnitude {args.magnitude_input}\tPhase {args.phase_input}")
print('Average PESQ score:', avg_pesq_score)
print('Average STOI score:', avg_stoi_score)

print (f"\t{args.magnitude_input}\t{args.phase_input}\t{avg_pesq_score}\t{avg_stoi_score}")
