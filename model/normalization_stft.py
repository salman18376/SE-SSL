import torch.nn as nn

class STFTNormaliztion(nn.Module):
    def __init__(self, eps=1e-5):
        super(STFTNormaliztion, self).__init__()
        self.eps = eps
        
        self.register_buffer('mean', None)  # register mean as a persistent buffer with initial value None
        self.register_buffer('std', None)  

    def forward(self, x):

        # x is a batch of STFT magnitudes with shape (batch_size, channels, freq_bins, time_frames)

        if self.mean is None or self.std is None:
            # compute the mean and standard deviation of the input along the frequency and time axes

            self.mean = x.mean(dim=(2, 3), keepdim=True)  
            self.std = x.std(dim=(2, 3), keepdim=True)  

        # normalize the input by subtracting the mean and dividing by the standard deviation
        x = (x - self.mean) / (self.std + self.eps)

        return x