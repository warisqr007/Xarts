import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm

from src.models.utils import init_weights, get_padding
from src.models.components.causal_conv import CausalConv
from src.models.components.causal_resblock import ResBlock
from src.models.components.fast_glu import FastGLU
from src.models.components.lookahead_block import LookaheadBlock

LRELU_SLOPE = 0.1


class HiFiEncoder(torch.nn.Module):
    def __init__(self, h):
        super(HiFiEncoder, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_downsamples = len(h.downsample_rates)
        self.up_buf_len = 1

        resblock = ResBlock
        
        self.conv_pre = CausalConv(1, h.downsample_initial_channel, 7, 1)

        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates, h.downsample_kernel_sizes)):
            self.downs.append(
                CausalConv(h.downsample_initial_channel * (2 ** i), 
                           h.downsample_initial_channel * (2 ** (i + 1)), 
                           k, u))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h.downsample_initial_channel * (2 ** i)
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, 0, d))

        self.conv_post = CausalConv(ch * 2, h.upsample_initial_channel, 7, 1)
        self.final_act = FastGLU(h.upsample_initial_channel)

        if h.lookahead > 0:
            self.lookahead = LookaheadBlock(h.upsample_initial_channel, h.lookahead)

        if hasattr(h, "context_layer_type"):
            if h.context_layer_type == 'transformer':
                self.context_layer = None
            elif h.context_layer_type == 'mamba':
                self.context_layer = None  
            else:
                print("No context layer")
    
        self.code_predictor = nn.Linear(h.upsample_initial_channel, h.num_codes)


    def init_buffers(self, batch_size, device):
        res_buf = []
        for i in range(self.num_downsamples):
            for j in range(self.num_kernels):
                ctx_buf = self.resblocks[i * self.num_kernels + j].init_ctx_buf(batch_size, device)
                res_buf.append(ctx_buf)
        down_buf = []
        for i, (u, k) in enumerate(zip(self.h.downsample_rates, self.h.downsample_kernel_sizes)):
            ctx_buf = self.downs[i].init_ctx_buf(batch_size, device)
            down_buf.append(ctx_buf)

        pre_conv_buf = self.conv_pre.init_ctx_buf(batch_size, device)
        post_conv_buf = self.conv_post.init_ctx_buf(batch_size, device)

        buffers = pre_conv_buf, res_buf, down_buf, post_conv_buf
        return buffers
    
    def forward(self, x, buffers=None):
        if buffers is None:
            buffers = self.init_buffers(x.size(0), x.device)
        pre_conv_buf, res_buf, down_buf, post_conv_buf = buffers
        #pre conv buff
        x, pre_conv_buf = self.conv_pre(x, pre_conv_buf)
        for i in range(self.num_downsamples):
            xs = None
            for j in range(self.num_kernels):
                ctx_buf = res_buf[i * self.num_kernels + j]
                if xs is None:
                    xs, ctx_buf = self.resblocks[i * self.num_kernels + j](x, ctx_buf)
                else:
                    xs_, ctx_buf = self.resblocks[i * self.num_kernels + j](x, ctx_buf)
                    xs += xs_
                res_buf[i * self.num_kernels + j] = ctx_buf
            x = xs / self.num_kernels

            #ctx buffer
            ctx_buf = down_buf[i]

            x = F.leaky_relu(x, LRELU_SLOPE)
            x, ctx_buf = self.downs[i](x, ctx_buf)
            down_buf[i] = ctx_buf

        x = F.leaky_relu(x)

        # post conv buff
        x, post_conv_buf = self.conv_post(x, post_conv_buf)
        # x = torch.tanh(x)
        # print("post conv shape", x.shape)
        z_embed = self.final_act(x.transpose(1, 2)).transpose(1, 2)

        # print("z_embed shape", z_embed.shape)
        if hasattr(self, 'lookahead'):
            z_embed = self.lookahead(z_embed)
            # print("lookahead z_embed shape", z_embed.shape)

        if hasattr(self, "context_layer"):
            z_embed = self.context_layer(z_embed)

        code = self.code_predictor(z_embed.transpose(1, 2)).transpose(1, 2)
        # print("code shape", code.shape)
        
        buffers = pre_conv_buf, res_buf, down_buf, post_conv_buf
        return code, z_embed, buffers

    def remove_weight_norm(self):
        for l in self.downs:
            l.remove_weight_norm()
        for l in self.resblocks:
            l.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()
        if hasattr(self, 'lookahead'):
            self.lookahead.remove_weight_norm()