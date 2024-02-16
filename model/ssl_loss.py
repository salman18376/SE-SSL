

import sys
import torch
import torch.nn as nn
from functools import partial
from geomloss import SamplesLoss
from transformers import AutoFeatureExtractor, AutoModel
from model.losses import weighted_sdr_loss

class PerceptualLoss(nn.Module):
    def __init__(self, model_type='wavlm', pretrained_model_name='microsoft/wavlm-large', sampling_rate=16000, padding="longest"):
        super().__init__()
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
        self.loss = weighted_sdr_loss
        # self.weighted_sdr_loss = weighted_sdr_loss
        if model_type == 'wavlm':
            self.processor =  AutoFeatureExtractor.from_pretrained(pretrained_model_name, sampling_rate=sampling_rate)
            self.model = AutoModel.from_pretrained(pretrained_model_name)
            self.model = self.model.to(self.device)
            # self.model = self.model.feature_extractor
            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False 
            
            self.model.eval()
        else:
            print('Please assign a valid model type')
            sys.exit()

    def forward(self, noisy, y_hat, y):
        with torch.no_grad():
            inputs = self.processor(y_hat, return_tensors="pt", sampling_rate = 16000).input_values.to(self.device)
            targets = self.processor(y, return_tensors="pt", sampling_rate = 16000).input_values.to(self.device)
            noisy = self.processor(noisy, return_tensors="pt", sampling_rate = 16000).input_values.to(self.device)
            inputs= inputs.squeeze(dim=0)
            targets= targets.squeeze(dim=0)
            noisy= noisy.squeeze(dim=0)
            
            y_hat_output = self.model(inputs).last_hidden_state #NOTE just undo hidden_state if want to use transformer
            y_output = self.model(targets).last_hidden_state
            noisy = self.model(noisy).last_hidden_state

        return self.loss(noisy,y_hat_output, y_output)