import os 
import torch
import torchaudio
import whisper

from transformers import AutoFeatureExtractor

from data.audio_processor import AudioProcessor


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, 
        noisy_path, #path to the folder containing noisy audio files
        clean_path, #path to the folder containing clean audio files
        n_fft, #window size for FFT calculation
        hop_length, #hop length for FFT calculation
        win_length, #window size for short time Fourier transform
        ssl_model_tag, #SSL model tag for feature extraction
        max_length_in_seconds = 10, #maximum length of audio files in seconds
        sampling_rate = 16000, #sampling rate for audio files
        padding = True, #whether to pad the signal with zeros
        truncate = True, #whether to truncate the signal
        randomly_sample_n_samples = -1,#32000, #number of samples to select for each audio file - if -1, all samples are selected
        normalize_waveform = True #whether to normalize the waveform
    ):
        self.filenames = sorted(os.listdir(noisy_path)) #list of file names in the noisy audio folder
        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.wavlm_feature_extractor = AutoFeatureExtractor.from_pretrained(ssl_model_tag) #creating an instance of the AutoFeatureExtractor class
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.ssl_model_tag = ssl_model_tag
        self.max_length_in_seconds = max_length_in_seconds if randomly_sample_n_samples == -1 else randomly_sample_n_samples/sampling_rate
        self.sampling_rate = sampling_rate
        self.padding = padding
        self.truncate = truncate
        self.randomly_sample_n_samples = randomly_sample_n_samples
        self.normalize_waveform = normalize_waveform

        self.audio_processor = AudioProcessor(
            sample_rate=sampling_rate,
            max_length_in_seconds=max_length_in_seconds if randomly_sample_n_samples == -1 else randomly_sample_n_samples/sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        self.epsilon = 1e-6 #small value to avoid division by zero

    def __len__(self): #returns the number of files in the dataset
        return len(self.filenames)

    
    # This function applies feature extraction on the waveform input using SSL model feature extraction
    def apply_feature_extractor(self, waveform):
        return self.wavlm_feature_extractor(
            waveform,
            padding="max_length" if self.padding else False,
            truncation=self.truncate,
            return_tensors="pt",
            max_length=int(self.max_length_in_seconds*self.wavlm_feature_extractor.sampling_rate),
            sampling_rate=self.sampling_rate,
            # return_attention_mask=None
        )
        
        

    def __getitem__(self, idx):
        # Get the filename for the current index
        filename = self.filenames[idx]
        # Load the clean and noisy versions of the waveform from their respective paths
        waveform_noisy, sr_noisy = torchaudio.load(os.path.join(self.noisy_path, filename))
        waveform_clean, sr_clean = torchaudio.load(os.path.join(self.clean_path, filename))
        
        if self.normalize_waveform:
            # normalization
            norm_factor = torch.sqrt(waveform_noisy.shape[1] / torch.sum(waveform_noisy ** 2))
            waveform_clean = waveform_clean * norm_factor
            waveform_noisy = waveform_noisy * norm_factor
            
        original_waveform_length = waveform_noisy.shape[1]


        if self.randomly_sample_n_samples > 0 and waveform_noisy.shape[1] > self.randomly_sample_n_samples:
            start_idx = torch.randint(0, waveform_noisy.shape[1] - self.randomly_sample_n_samples, (1,)).item()
            waveform_noisy = waveform_noisy[:, start_idx: start_idx + self.randomly_sample_n_samples]
            waveform_clean = waveform_clean[:, start_idx: start_idx + self.randomly_sample_n_samples]
            original_waveform_length = self.randomly_sample_n_samples
        
        
        
        # Get transformed versions of the clean and noisy waveforms (if requested)
        waveform_clean_transformed = waveform_clean
        waveform_noisy_transformed = waveform_noisy

        # Pad the waveforms (if requested)
        if self.padding:
            waveform_noisy_transformed = self.audio_processor._right_pad_if_necessary(waveform_noisy_transformed)
            waveform_clean_transformed = self.audio_processor._right_pad_if_necessary(waveform_clean_transformed)
         # If noise is truncated (if requested)    
        if self.truncate:
            waveform_noisy_transformed = self.audio_processor._cut_if_necessary(waveform_noisy_transformed)
            waveform_clean_transformed = self.audio_processor._cut_if_necessary(waveform_clean_transformed)
        
        # Get the magnitude and phase of the STFT of the clean and noisy waveforms
        mag_noisy, phase_noisy, complex_noisy, real_noisy , imag_noisy = self.audio_processor.get_spectrogram_from_waveform(waveform_noisy_transformed)
        mag_clean, phase_clean,  complex_clean, real_clean , imag_clean = self.audio_processor.get_spectrogram_from_waveform(waveform_clean_transformed)

        mag_noisy = mag_noisy + self.epsilon # for log1p compression and avoid numerical instability
        mag_clean = mag_clean + self.epsilon # for log1p compression and avoid numerical instability
        
        # Squeeze the waveforms
        waveform_clean = waveform_clean.squeeze()
        waveform_noisy = waveform_noisy.squeeze()
        input_waveform = self.apply_feature_extractor(waveform_noisy)
        # print(input_waveform.keys())  # Check the available keys in the feature_dict

        output_waveform = self.apply_feature_extractor(waveform_clean)

        item = {
            "input_stft": mag_noisy,
            "output_stft": mag_clean,
            "input_phase": phase_noisy,
            "output_phase": phase_clean,
            "input_complex": complex_noisy,
            "output_complex": complex_clean,
            "input_real": real_noisy,
            "input_imag": imag_noisy,
            "output_real": real_clean,
            "output_imag": imag_clean,
            "input_waveform": input_waveform["input_values"][0],
            "output_waveform": output_waveform["input_values"][0],
            "original_waveform_length": torch.tensor(original_waveform_length),

        }

        return item
    
    
    