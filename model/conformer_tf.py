import torch
import torch.nn as nn
from conformer import ConformerBlock


class ConformerTFLayer(nn.Module):
    '''
    This class implements the Conformer Time-Frequency layer.
    It consist of 2 Conformer blocks, one for attention on the Time axis and one for attention on the Frequency axis.
    '''
    def __init__(
        self,
        hidden_channels = 64,
        n_heads = 4,
        dropout = 0.1,
    ):
        '''
        :param hidden_channels: int. Dimension of the hidden layers.
        '''

        super(ConformerTFLayer, self).__init__()

        self.hidden_channels = hidden_channels
        
        self.n_heads = n_heads
        self.dropout = dropout

        self.one_to_n = nn.Linear(1, self.hidden_channels)

        self.n_to_one = nn.Linear(self.hidden_channels, 1)
    
        self.time_block = ConformerBlock(
            dim = self.hidden_channels,
            dim_head = self.hidden_channels // self.n_heads,
            heads = self.n_heads,
            conv_kernel_size = 31,
            attn_dropout = self.dropout,
            ff_dropout = self.dropout,
            conv_dropout = self.dropout,
        )

        self.freq_block = ConformerBlock(
            dim = self.hidden_channels,
            dim_head = self.hidden_channels // self.n_heads,
            heads = self.n_heads,
            conv_kernel_size = 31,
            attn_dropout = self.dropout,
            ff_dropout = self.dropout,
            conv_dropout = self.dropout,
        )

    def forward(self, x):

        B, T, F = x.shape
        # B, T, F -> B, T, F, 1
        x = x.unsqueeze(dim=-1)
        # B, T, F, 1 -> B, T, F, hidden_channels
        x = self.one_to_n(x)
        # B, T, F, hidden_channels -> BxF, T, hidden_channels
        x = x.permute(0, 2, 1, 3).reshape(B*F, T, self.hidden_channels)
        # BxF, T, hidden_channels -> BxF, T, hidden_channels
        x = self.time_block(x) + x
        # BxF, T, hidden_channels -> B, F, T, hidden_channels
        x = x.reshape(B, F, T, self.hidden_channels)
        # B, F, T, hidden_channels -> B, T, F, hidden_channels
        x = x.permute(0, 2, 1, 3)
        # B, T, F, hidden_channels -> BxT, F, hidden_channels
        x = x.reshape(B*T, F, self.hidden_channels)
        # BxT, F, hidden_channels -> BxT, F, hidden_channels
        x = self.freq_block(x) + x
        # BxT, F, hidden_channels -> B, T, F, hidden_channels
        x = x.reshape(B, T, F, self.hidden_channels)
        # B, T, F, hidden_channels -> B, T, F, 1
        x = self.n_to_one(x)
        # B, T, F, 1 -> B, T, F
        x = x.squeeze(dim=-1)

        return x

class ConformerTimeFrequencyHead(nn.Module):

    def __init__(
        self,
        input_dim, # ssl_dim + stft_dim = 768 + 201 = 969
        hidden_dim, # arbitrary - usually 768
        output_dim, # stft_dim = 201
        n_heads = 12,
        dim_feedforward = 3072, # not used
        num_layers = 2,
        dropout = 0.1, # can be changed for regularization
        add_positional_encoding = True,
        max_len = 1000,
    ):
        '''
        :param input_dim: int. Dimension of the input.
        :param hidden_dim: int. Dimension of the hidden layers.
        :param output_dim: int. Dimension of the output.
        :param num_heads: int. Number of heads in the multihead attention.
        :param dim_feedforward: int. Dimension of the feedforward layers.
        :param num_layers: int. Number of layers in the Conformer.
        :param dropout: float. Dropout value.
        :param add_positional_encoding: bool. Whether to add positional encoding.
        :param max_len: int. Maximum length of the input.
        '''

        super(ConformerTimeFrequencyHead, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward # not used
        self.num_layers = num_layers
        self.dropout = dropout
        self.add_positional_encoding = add_positional_encoding
        self.max_len = max_len


        
        self.in_l = nn.Linear(input_dim, hidden_dim)
        if add_positional_encoding:
            self.positional_encoding = nn.Embedding(max_len, hidden_dim)
        self.conformer_tf = nn.Sequential()
        for i in range(self.num_layers):
            self.conformer_tf.add_module(
                f"conformer_tf_{i}",
                ConformerTFLayer(
                    hidden_channels = self.hidden_dim,
                    n_heads = self.n_heads,
                    dropout = self.dropout,
                )
            )

        self.out_l = nn.Linear(hidden_dim, output_dim)

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
        if self.add_positional_encoding:
            x = x + self.positional_encoding(torch.arange(x.shape[1], device=x.device))
        x = self.conformer_tf(x)
        x = self.out_l(x)

        return x