import torch
import torch.nn as nn
from model.learnable_sigmoid import LearnableSigmoid    


class LSTM(nn.Module):

    def __init__(
        self,
        input_dim, # ssl_dim + stft_dim = 768 + 201 = 969
        hidden_dim, # arbitrary - usually 256
        output_dim, # stft_dim = 201
        input_channels = 1,
        num_layers = 2,
        dropout = 0.1, # can be changed for regularization
    ):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.type = type

        self.in_l = nn.Linear(input_dim, hidden_dim, bias=True)
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True, 
            num_layers=num_layers, 
            dropout=dropout,
        )
        #change hidden_dim -> hidden_dim * 2 if removing the line from the paper code
        self.out_l = nn.Linear(hidden_dim, output_dim, bias=True) 

    def forward(self, x):
        x = x.squeeze(dim=1) # bs, seq_len, feat_dim
        x = self.in_l(x)
        x, _ = self.lstm(x)
        x = x[:,:,:int(x.size(-1)/2)]+x[:,:,int(x.size(-1)/2):] # from the code of the paper
        x = self.out_l(x)
        return x # bs, seq_len, feat_dim
