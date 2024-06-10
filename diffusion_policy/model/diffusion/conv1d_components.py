import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        ###在该参数下，每次downsample后，通道数不变，序列长度缩减一半
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #一维转置卷积层
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            #这里的padding在kernal_size为奇数时，等于padding='same'
            # nn.Conv1d(inp_channels, out_channels, kernel_size, padding='same')
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            #Mish函数在深度神经网络中表现出色，因为它有助于缓解梯度消失问题，加速收敛速度，并提高模型的泛化能力
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1,256,16))
    o = cb(x)
