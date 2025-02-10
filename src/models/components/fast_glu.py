import torch.nn as nn


class FastGLU(nn.Module):
    def __init__(self, in_dim):
        super(FastGLU, self).__init__()

        self.in_dim = in_dim
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(in_dim, 2*in_dim)

    def forward(self, x):
        out = self.fc(x)
        return out[..., :self.in_dim] * self.sigmoid(out[..., self.in_dim:])