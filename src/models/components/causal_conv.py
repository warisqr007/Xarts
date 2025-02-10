import torch
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from src.models.utils import init_weights


class CausalConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(CausalConv, self).__init__()
        self.in_channel = in_channel
        self.buf_len = kernel_size - 1
        self.layer = weight_norm(Conv1d(in_channel, out_channel, kernel_size, stride, padding))

    def init_ctx_buf(self, batch_size, device):
        ctx_buf = torch.zeros(
            (batch_size, self.in_channel, self.buf_len), device=device)
        return ctx_buf
    
    def forward(self, x, ctx_buf):
        x = torch.cat((ctx_buf, x), dim=-1)

        if self.buf_len!=0:
            ctx_buf = x[..., -self.buf_len:]

        x = self.layer(x)
        return x, ctx_buf
    
    def remove_weight_norm(self):
        remove_weight_norm(self.layer)


class CausalConvTranspose1D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(CausalConvTranspose1D, self).__init__()
        self.in_channel = in_channel
        self.buf_len = kernel_size - 1
        self.upsample_factor = stride
        self.layer = weight_norm(ConvTranspose1d(in_channel, out_channel, kernel_size, stride, padding))

    def init_ctx_buf(self, batch_size, device):
        ctx_buf = torch.zeros(
            (batch_size, self.in_channel, self.buf_len), device=device)
        return ctx_buf
    
    def forward(self, x, ctx_buf):
        # x = torch.cat((ctx_buf, x), dim=-1)
        # ctx_buf = x[..., -self.buf_len:]
        n = x.shape[-1]
        x = self.layer(x)
        x = x[..., :(n * self.upsample_factor)]
        return x, ctx_buf
    
    def init_weights(self):
        self.layer.apply(init_weights)

    def remove_weight_norm(self):
        remove_weight_norm(self.layer)