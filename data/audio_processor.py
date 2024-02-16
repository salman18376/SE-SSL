import torch
import numpy as np

class AudioProcessor:
    def __init__(       
            self, 
            sample_rate, 
            max_length_in_seconds,
            n_fft, 
            hop_length, 
            win_length, 
            normalized_stft = False,
        ):
        self.sample_rate = sample_rate
        self.max_length_in_seconds = max_length_in_seconds
        self.target_length = sample_rate * max_length_in_seconds
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized_stft = normalized_stft
    
    def _cut_if_necessary(self, signal, target_length=None):
        target_length = target_length if target_length else self.target_length
        if signal.shape[1] > self.target_length:
            signal = signal[:, :self.target_length]
        return signal

    def _right_pad_if_necessary(self, signal, target_length=None):
        
        is_1d = len(signal.shape) == 1
        if is_1d:
            signal = signal.unsqueeze(0)

        target_length = target_length if target_length else self.target_length
        length_signal = signal.shape[1]
        if length_signal < target_length:
            num_missing_samples = target_length - length_signal
            last_dim_padding = (0, int(num_missing_samples))
            signal = torch.nn.functional.pad(signal, last_dim_padding)

        if is_1d:
            signal = signal.squeeze(0)

        return signal
    
    
    def get_spectrogram_from_waveform(self, signal):
        '''
        This function computes the magnitude and phase of the spectrogram using the STFT.
        :param signal: tensor of shape (1, n_samples) or (bs, 1, n_samples)
        :return: magnitude and phase of the spectrogram.
        '''            
        device = signal.device
        
        
        stft = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            return_complex=True,
            window=torch.hamming_window(self.win_length ).to(device),
            normalized=self.normalized_stft,
            # pad_mode='constant',  # Specify the desired pad_mode
            # onesided=True,
            
        )

        mag = torch.abs(stft)

        mag = mag.squeeze(dim=0)
        if len(mag.shape) == 3: mag = mag.permute(0, 2, 1)
        else: mag = mag.permute(1, 0)
        
        
        phase = torch.angle(stft)
        
        phase = phase.squeeze(dim=0)
        if len(phase.shape) == 2: phase = phase.permute(1, 0)
    
        else: phase = phase.permute(0, 2, 1)
        
        complex = torch.stack((mag*torch.cos(phase), mag*torch.sin(phase)), dim=-1)
        
        real = complex[..., 0]
        imag = complex[..., 1]
        
        complex = complex.squeeze(dim=0)

        if len(complex.shape) == 3: complex = complex.permute(1, 0, 2)
        else: complex = complex.permute(0, 2, 1, 3)

        return mag, phase , complex , real , imag

    def get_waveform_from_spectrogram(self, mag, phase, target_length=None):
        '''
        This function computes the waveform from the magnitude and phase of the spectrogram using the ISTFT.
        :param mag: magnitude of the spectrogram.
        :param phase: phase of the spectrogram.
        :return: waveform.
        '''

        real = mag * torch.cos(phase).squeeze(dim=1)
        imag = mag * torch.sin(phase).squeeze(dim=1)

        # create complex tensor for istft
        complex_spec = torch.complex(real, imag)
        complex_spec = complex_spec.squeeze(dim=0)
        if len(complex_spec.shape) == 3: complex_spec = complex_spec.permute(0, 2, 1)
        else: complex_spec = complex_spec.permute(1, 0)
        
        # get the device of the spectrogram
        device = complex_spec.device
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hamming_window(self.win_length).to(device),
            center=False,
            length=target_length if target_length else None,
            normalized=self.normalized_stft,
            # onesided=True
            
        )
        

        return waveform
        
    def complex_consistency(self,predicted_complex, target_length=None):
        
        # Extract the real and imaginary parts
        real = predicted_complex[:, :, :, 0]
        imag = predicted_complex[:, :, :, 1]
        
        # Create complex tensor for ISTFT
        complex_spec = torch.complex(real, imag)
        complex_spec = complex_spec.squeeze(dim=0)
        
        # Permute dimensions if needed
        if len(complex_spec.shape) == 3:
            complex_spec = complex_spec.permute(0, 2, 1)
        else:
            complex_spec = complex_spec.permute(1, 0)
            
        
        # Get the device of the spectrogram
        device = complex_spec.device
        
        # Perform ISTFT
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hamming_window(self.win_length).to(device),
            center=False,
            length=target_length if target_length else None,
            normalized=self.normalized_stft,
            
        )
        
        
        stft = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=False,
                return_complex=True,
                window=torch.hamming_window(self.win_length ).to(device),
                normalized=self.normalized_stft,
                pad_mode='constant',  # Specify the desired pad_mode
                # onesided=True,
                
            )
        mag = torch.abs(stft)
        
        mag = mag.squeeze(dim=0)
        if len(mag.shape) == 3: mag = mag.permute(0, 2, 1)
        else: mag = mag.permute(1, 0)
        

        phase = torch.angle(stft)
        
        phase = phase.squeeze(dim=0)
        if len(phase.shape) == 2: phase = phase.permute(1, 0)

        else: phase = phase.permute(0, 2, 1)

        complex = torch.stack((mag*torch.cos(phase), mag*torch.sin(phase)), dim=-1)
        
        real = complex[..., 0]
        imag = complex[..., 1]
        
        complex = complex.squeeze(dim=0)

        if len(complex.shape) == 3: complex = complex.permute(1, 0, 2)
        else: complex = complex#.permute(0, 2, 1, 3)
        return  complex 
    
    def get_phase_from_waveform(self, signal):
        '''
        This function computes the magnitude and phase of the spectrogram using the STFT.
        :param signal: tensor of shape (1, n_samples) or (bs, 1, n_samples)
        :return: magnitude and phase of the spectrogram.
        '''            
        device = signal.device
        
        
        stft = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            return_complex=True,
            window=torch.hann_window(self.win_length ).to(device),
            normalized=self.normalized_stft,
            # pad_mode='constant',  # Specify the desired pad_mode
            # onesided=True,
            
        )

              
        phase = torch.angle(stft)
        
        phase = phase.squeeze(dim=0)
        if len(phase.shape) == 2: phase = phase.permute(1, 0)
    
        else: phase = phase.permute(0, 2, 1)
        
        return phase