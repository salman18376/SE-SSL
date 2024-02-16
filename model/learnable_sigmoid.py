import torch
import torch.nn as nn


class LearnableSigmoid(nn.Module):
    """
    This class implements a learnable sigmoid function.
    Alpha and beta are both learnable parameters, initialized to the specified values.
    :param in_features: The number of input features.
    :param alpha: The alpha parameter, multiplied to the input.
    :param beta: The beta parameter, multiplied to the sigmoid output.
    """

    def __init__(self, in_features=201, alpha=1.0, beta=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
        self.beta = nn.Parameter(torch.ones(in_features) * beta)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        #return self.beta * torch.sigmoid(self.alpha * x-1)
        return self.beta / (1 + torch.exp(1-self.alpha * x))