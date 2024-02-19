import os

import torchaudio
from transformers import Wav2Vec2Model
from typing import Union
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from data.audio_processor import AudioProcessor
from data.compressor import SpectrogramCompressor
from model.lstm import LSTM
from model.unet import UNet
from model.transformer import TransformerHead
from model.conformer import ConformerHead
from model.learnable_sigmoid import LearnableSigmoid
from model.conformer_tf import ConformerTimeFrequencyHead
import matplotlib.pyplot as plt
import numpy as np



class SpeechEnhancementModel(nn.Module):
    '''
    This class implements the Speech Enhancement Model.
    It leverages an SSL model to extract features from the noisy waveform.
    Then, it uses two different heads to predict the magnitude and phase of the STFT.
    '''
    
    def __init__(
        self, 
        ssl_model, 
        freeze_ssl=False, 
        magnitude_head="cnn", 
        phase_head="cnn",
        complex_head="cnn",
        ssl_embedding_dim=768, 
        stft_embedding_dim=201,
        type="masking",
        compressor=None,
        s3prl=False,
        sigmoid_type="learnable",
        use_all_layers=False,
    ):
        '''
        :param ssl_model: The SSL model to be used for feature extraction. It is expected to return a last_hidden_state.
        :param freeze_ssl: Whether to freeze the SSL model during training or not.
        :param magnitude_head: The type of head to be used for the magnitude prediction (cnn, lstm, unet, transformer).
        :param phase_head: The type of head to be used for the phase prediction (cnn, lstm, unet, transformer).
        :param ssl_embedding_dim: The dimension of the SSL model output.
        :param stft_embedding_dim: The dimension of the STFT embedding.
        :param type: The type of model (masking, mapping). Masking predict a mask to be applied to the input STFT. Mapping predict the STFT directly.
        :param sigmoid_type: The type of sigmoid to be used for the masking model (learnable, standard).
        :param compressor: The compressor to be used to compress the STFT. It is expected to have a compress and decompress method.
        :param use_all_layers: Whether to use all the layers of the SSL model or just the last one. 
            If True, the model will use all the layers and perform a weighted sum of the outputs.
        '''
        super(SpeechEnhancementModel, self).__init__()

        self.ssl_model = ssl_model
        self.freeze_ssl = freeze_ssl
        self.magnitude_head = magnitude_head
        self.phase_head = phase_head
        self.complex_head = complex_head
        self.ssl_embedding_dim = ssl_embedding_dim
        self.stft_embedding_dim = stft_embedding_dim
        self.type = type
        self.s3prl = s3prl
        self.compressor = compressor
        # this is to check if the STFT is 2x the size of the wav2vec2_hidden_state
        self.n_frames_tolerance = 16 
        
        if self.magnitude_head is not None:
            print(f"Initializing magnitude head of type {self.magnitude_head}") 
            self.model_head_mag = self.get_head(self.ssl_embedding_dim + self.stft_embedding_dim, self.magnitude_head)
        else:
            self.model_head_mag = None
        if self.phase_head is not None:
            print(f"Initializing phase head of type {self.phase_head}")
            self.head_input_dim = self.stft_embedding_dim * 2
            # self.head_input_dim = self.stft_embedding_dim
            self.model_head_phase = self.get_head(self.head_input_dim, self.phase_head)
        else:
            self.model_head_phase = None
        
        if self.complex_head is not None:
            print(f"Initializing complex head of type {self.complex_head}")
            self.head_input_dim = self.stft_embedding_dim * 2
            # self.head_input_dim = self.stft_embedding_dim
            self.model_head_complex = self.get_head(self.head_input_dim, self.complex_head)
        else:
            self.model_head_complex = None
        
        

        if freeze_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
            self.ssl_model.eval()



        if self.type == "masking":
            if sigmoid_type == "learnable":
                #self.sigmoid = LearnableSigmoid(self.stft_embedding_dim)
                self.magnitude_sigmoid = LearnableSigmoid(self.stft_embedding_dim)
                self.complex_sigmoid = LearnableSigmoid(self.stft_embedding_dim)
                self.phase_sigmoid = LearnableSigmoid(self.stft_embedding_dim)
            elif sigmoid_type == "standard":
                self.magnitude_sigmoid = nn.Sigmoid()
                self.complex_sigmoid = nn.Sigmoid()
                self.phase_sigmoid = nn.Sigmoid()
            else:
                raise ValueError(f"Sigmoid type {sigmoid_type} not supported.")
            
        
        self.use_all_layers = use_all_layers
        if self.use_all_layers:
            self.softmax = nn.Softmax(-1)
            self.layer_weights = nn.Parameter(torch.ones(self.ssl_model.config.num_hidden_layers))
            self.layer_weights.requires_grad = True
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.ssl_embedding_dim) for _ in range(self.ssl_model.config.num_hidden_layers)])
            self.layer_norms.requires_grad = True
        
    def get_head(self, input_dim, head_type):
        '''
        This function returns a head to be used on top of the model.
        :param input_dim: The input dimension of the head.
        :param head_type: The type of the head (cnn, lstm, unet, transformer).
        :return: The nn.Module head.
        '''

        if head_type == "cnn":
            return CNN(input_dim=input_dim, output_dim=self.stft_embedding_dim, input_channels=1,
                layers=[
                    (32, 3, 1),
                    (64, 3, 1),
                    (128, 3, 1),
                    (64, 3, 1),
                    (32, 3, 1),
                    (1, 3, 1),
                ]
            )
        elif head_type == "lstm":
            return LSTM(input_dim=input_dim, hidden_dim=256, output_dim=self.stft_embedding_dim, input_channels=1, num_layers=2)
        
        elif head_type == "unet":
            return UNet(input_dim=input_dim, n_classes=1, input_channels=1, bilinear=True)
        
        elif head_type == "transformer":
            return TransformerHead(
                input_dim=input_dim, hidden_dim=self.ssl_model.config.hidden_size,
                output_dim=self.stft_embedding_dim, n_heads=self.ssl_model.config.num_attention_heads,
             num_layers=2,
                dropout=0.1
            )
        
        elif head_type == "conformer": 
             return ConformerHead(
                input_dim=input_dim, hidden_dim=self.ssl_model.config.hidden_size,
                output_dim=self.stft_embedding_dim, n_heads=self.ssl_model.config.num_attention_heads,
                num_layers=2,
                dropout=0.1
            )
            
        else:
            raise ValueError(f"Head type {head_type} not supported.")
    

    def forward(self, batch):
        '''
        This function performs a forward pass of the model.
        :param batch: The batch to be used for the forward pass.
        :return: The estimated magnitude and phase of the STFT for the "clean" speech.
        '''
        complex_noisy = batch["input_complex"]
        complex_clean =batch["output_complex"]

        
        complex_clean = complex_clean.permute(0, 2, 1, 3)
        complex_noisy = complex_noisy.permute(0, 2, 1, 3)
        # ---------------------- SSL ----------------------
        if self.use_all_layers:
            wav2vec2_hidden_states = self.ssl_model(input_values = batch["input_waveform"], return_dict=True).hidden_states
            if wav2vec2_hidden_states[0].shape[1] > self.ssl_model.config.num_hidden_layers: # first is the CNN embedding layer
                wav2vec2_hidden_states = wav2vec2_hidden_states[1:]
            wav2vec2_hidden_state = torch.zeros_like(wav2vec2_hidden_states[0])
            
            weights = self.softmax(self.layer_weights)
            for i in range(self.ssl_model.config.num_hidden_layers):
                hidden_state = wav2vec2_hidden_states[i]
                hidden_state = self.layer_norms[i](hidden_state)
                wav2vec2_hidden_state += hidden_state * weights[i]
                
        
            
        else:                                                      
            
            wav2vec2_hidden_state = self.ssl_model(batch["input_waveform"], return_dict=True).last_hidden_state

        # check if STFT 2x the size of the wav2vec2_hidden_state -- if yes, duplicate the wav2vec2_hidden_state
        if batch["input_stft"].shape[1] >= wav2vec2_hidden_state.shape[1]*2-self.n_frames_tolerance:
            batch_size, time_dim, embed_dim = wav2vec2_hidden_state.shape
            wav2vec2_hidden_state = wav2vec2_hidden_state.repeat(1,1,2).reshape(batch_size,-1,embed_dim)
            wav2vec2_hidden_state = wav2vec2_hidden_state[:, :batch["input_stft"].shape[1], :]
        combined_representation = torch.cat(
            (wav2vec2_hidden_state, batch["input_stft"][..., :wav2vec2_hidden_state.shape[1], :].to(wav2vec2_hidden_state)), 
            dim=-1
        ) #([32, 798, 201]) + ([32, 798, 768]) = ([32, 798, 969])
        combined_representation = combined_representation.unsqueeze(dim=0)
        combined_representation = combined_representation.permute(1, 0, 2, 3)
        
        # ---------------------- Magnitude ----------------------
        if self.model_head_mag is not None:
            
            output_mag = self.model_head_mag(combined_representation) 
            if self.type == "masking": output_mag = self.magnitude_sigmoid(output_mag)
            
        output_mag_phase = output_mag * batch["input_stft"]
            
        
        
        # ------------------------ Phase ------------------------
        if self.model_head_phase is not None:

            output_phase = self.model_head_phase(complex_noisy) #([32, 798, 201,2])
            
            if self.type == "masking": 
                output_phase = output_phase 
    
        else:
            output_phase = batch["input_phase"]
            
         # ------------------------ complex domain ------------------------
            
        if self.model_head_complex is not None:
            
            
            output_complex = self.model_head_complex(complex_noisy) 
            if self.type == "masking": output_complex = self.complex_sigmoid(output_complex)#([32, 798, 201])
        else:
            output_complex = complex_noisy
        
       
        
        return output_mag, output_phase
