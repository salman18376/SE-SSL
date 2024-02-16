import torch
import torch.nn as nn


class SpectrogramCompressor(nn.Module):
    def __init__(
            self, 
            alpha=0.3,
            learnable=False, 
            compressor_type="log1p",
            stft_dim=201
        ):
        '''
        :param alpha: exponential value used when compressor_type = "exponential"
        :param learnable: whether alpha and beta are learnable parameters
        :param compressor_type: log1p or exponential, the type of compression to use
        '''
        super(SpectrogramCompressor, self).__init__()
        self.alpha = alpha
        self.learnable = learnable
        self.compressor_type = compressor_type

        if self.learnable and compressor_type == "exponential":
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def compress(self, mag):
        if self.compressor_type == "log1p":
            mag = torch.log1p(mag)
        elif self.compressor_type == "exponential":
            mag = mag ** self.alpha
        else:
            raise ValueError(f"{self.compressor_type} is not supported as compression type.")

        return mag

    def decompress(self, mag):
        if self.compressor_type == "log1p":
            mag = torch.expm1(mag)
        elif self.compressor_type == "exponential":
            mag = mag ** (1/self.alpha)
        else:
            raise ValueError(f"{self.compressor_type} is not supported as compression type.")

        return mag