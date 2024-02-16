import torch
import torch.nn as nn
from conformer import ConformerBlock


        
class ConformerHead(nn.Module):

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
        '''
        :param input_dim: int. Dimension of the input.
        :param hidden_dim: int. Dimension of the hidden layers.
        :param output_dim: int. Dimension of the output.
        :param n_heads: int. Number of heads in the multihead attention.
        :param dim_feedforward: int. Dimension of the feedforward layers.
        :param num_layers: int. Number of layers in the Conformer.
        :param dropout: float. Dropout value.
        '''

        super(ConformerHead, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # self.pre_norm = nn.LayerNorm(input_dim)
        self.in_l = nn.Linear(input_dim, hidden_dim)
        
    
        self.conformer = nn.Sequential()
        for i in range(self.num_layers):
            self.conformer.add_module(
                f"conformer_{i}",
                ConformerBlock(
                    dim = self.hidden_dim,
                    dim_head = self.hidden_dim // n_heads,
                    heads = n_heads,
                    ff_mult = dim_feedforward // self.hidden_dim,
                    conv_expansion_factor = 2,
                    conv_kernel_size = 32,
                    attn_dropout = self.dropout,
                    ff_dropout = self.dropout,
                    conv_dropout = self.dropout,
                )
            )
        self.out_l = nn.Linear(hidden_dim, output_dim)
        self.post_norm = nn.LayerNorm(output_dim)

    def init_weights(self):
        '''
        This function initializes the weights of the model.
        '''
        run_init_on = ["conv", "linear"]
        constant_init = ["batchnorm", "norm", "bias"]

        for name, param in self.named_parameters():
            if any([name in name_init for name_init in run_init_on]):
                if any([name_init in name for name_init in constant_init]):
                    nn.init.constant_(param, 0.0)
                else:
                    nn.init.xavier_uniform_(param)


    def forward(self, x):
            
            if len(x.shape) == 4:
                x = x.squeeze(dim=1)
            x = self.in_l(x)
            x = self.conformer(x)  
            x = self.out_l(x) 
            
            x = self.post_norm(x)
            # print(x.shape)
            return x