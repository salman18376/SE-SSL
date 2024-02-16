import torch
import torch.nn as nn
from model.learnable_sigmoid import LearnableSigmoid 


class TransformerHead(nn.Module):

    def __init__(
        self,
        input_dim, # ssl_dim + stft_dim = 768 + 201 = 969
        hidden_dim, # arbitrary - usually 768
        output_dim, # stft_dim = 201
        n_heads = 12,
        dim_feedforward = 3072,
        num_layers = 2,
        dropout = 0.1, # can be changed for regularization
    ):
        super(TransformerHead, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.type = type

        self.in_l = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Sequential()
        for i in range(self.num_layers):
            self.transformer.add_module(
                f"transformer_{i}",
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim, 
                    nhead=n_heads, 
                    dim_feedforward=dim_feedforward, 
                    dropout=self.dropout,
                    batch_first=True
                )
            )
        self.out_l = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):

        if len(x.shape) == 4:
            x = x.squeeze(dim=1)
       
        x = self.in_l(x)
        x = self.transformer(x)
        x = self.out_l(x)
            
        return x
