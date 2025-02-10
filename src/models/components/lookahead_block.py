import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

from src.models.utils import get_padding


class LookaheadBlock(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super(LookaheadBlock, self).__init__()
        self.in_channel = in_channel
        self.layer = weight_norm(nn.Conv1d(in_channel, in_channel, kernel_size, 1, get_padding(kernel_size, 1)))
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(in_channel)
    
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return x
    
    def remove_weight_norm(self):
        remove_weight_norm(self.layer)