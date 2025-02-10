import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

from src.models.utils import init_weights

LRELU_SLOPE = 0.1


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, padding=0, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h

        self.channels = channels
        self.kernel_size = kernel_size
        num_layers = 3
        self.sum_dilations = dilation[0] + dilation[1] + dilation[2]
        # Compute buffer lengths for each layer
        # buf_length[i] = (kernel_size - 1) * dilation[i]
        self.buf_lengths = [(kernel_size - 1) * dilation[i]
                            for i in range(num_layers)]

        # Compute buffer start indices for each layer
        self.buf_indices = [0]
        for i in range(num_layers - 1):
            self.buf_indices.append(
                self.buf_indices[-1] + self.buf_lengths[i])
            
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=padding)), 
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=padding)), 
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=padding))])
        self.convs1.apply(init_weights)

        # buf_length[i] = (kernel_size - 1) * dilation[i]
        self.buf_lengths_conv2 = [(kernel_size - 1) for i in range(num_layers)]

        # Compute buffer start indices for each layer
        self.buf_indices_conv2 = [0]
        for i in range(num_layers - 1):
            self.buf_indices_conv2.append(
                self.buf_indices_conv2[-1] + self.buf_lengths_conv2[i])

        self.convs2 = nn.ModuleList(
            [weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=padding)),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=padding)),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=padding))])
        self.convs2.apply(init_weights)

    def init_ctx_buf(self, batch_size, device):
        """
        Returns an initialized context buffer for a given batch size.
        """
        ctx_buf_o = torch.zeros((batch_size, self.channels, (self.kernel_size - 1) * (self.sum_dilations)), device=device)
        ctx_buf_t = torch.zeros((batch_size, self.channels, (self.kernel_size - 1) * 3), device=device)
        ctx_buf = ctx_buf_o, ctx_buf_t
        return ctx_buf
    
    def forward(self, x, ctx_buf):
        ctx_buf_o, ctx_buf_t = ctx_buf
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            #input: concatenation of current output and context
            dcc_in = torch.cat(
                (ctx_buf_o[..., buf_start_idx:buf_end_idx], x), dim=-1)

            # Push current output to the context buffer
            ctx_buf_o[..., buf_start_idx:buf_end_idx] = \
                dcc_in[..., -self.buf_lengths[i]:]

            xt = F.leaky_relu(dcc_in, LRELU_SLOPE)
            xt = c1(xt)

            buf_start_idx = self.buf_indices_conv2[i]
            buf_end_idx = self.buf_indices_conv2[i] + self.buf_lengths_conv2[i]

            #input: concatenation of current output and context
            dcc_in = torch.cat(
                (ctx_buf_t[..., buf_start_idx:buf_end_idx], xt), dim=-1)

            # Push current output to the context buffer
            ctx_buf_t[..., buf_start_idx:buf_end_idx] = \
                dcc_in[..., -self.buf_lengths_conv2[i]:]

            xt = F.leaky_relu(dcc_in, LRELU_SLOPE)
            xt = c2(xt)

            # Residual connection
            x = xt + x
        ctx_buf = ctx_buf_o, ctx_buf_t
        return x, ctx_buf

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)