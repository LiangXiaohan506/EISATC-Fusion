'''FB-Sinc-CSANet from Jiaming Chen et al 2023.
See details at https://iopscience.iop.org/article/10.1088/1741-2552/acbb2c

    Notes
    -----
    The initial values in this model are based on the values identified by the authors

    References
    ----------
    Chen J, Wang D, Yi W, et al. 
    Filter bank sinc-convolutional network with channel self-attention for high performance motor imagery decoding[J]. 
    Journal of Neural Engineering, 2023, 20(2): 026001.
'''

import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import torch.nn as nn
from torchinfo import summary
from torchstat import stat
from utils.SincConv_util import SincConv2d
from utils.util import Conv2dWithConstraint, LinearWithConstraint


#%%
class CSA(nn.Module):
    def __init__(self):
        super(CSA, self).__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        shape = x.shape # (batch, C, H, W)
        v = torch.reshape(x, (shape[0], shape[1], -1)) # (batch, C, H*W)
        k = torch.reshape(x, (shape[0], shape[1], -1)) # (batch, C, H*W)
        q = torch.reshape(x, (shape[0], shape[1], -1)) # (batch, C, H*W)
        k = torch.transpose(k, dim0=1, dim1=2) # (batch, H*W, C)
        qk = torch.matmul(q, k) # (batch, C, C)
        qk = self.softmax(qk)
        qkv = torch.matmul(qk, v) # (batch, C, H*W)
        qkv = torch.reshape(qkv, (shape[0], shape[1], shape[2], shape[3])) # (batch, C, H, W)
        out = qkv + x

        return out


#%%
class FB_Sinc_CSANet(nn.Module):
    def __init__(self, F1=32, L=65, D=2, poolFB=4, samRat=250, samples=1000, channels=22, n_classes=4, droRate_FB=0.3, dropoutRate=0.3):
        super(FB_Sinc_CSANet, self).__init__()

        self.filterBank_θ = nn.Sequential(
            SincConv2d(
                in_channels=1, 
                out_channels=F1,
                kernel_size=L,
                sample_rate=samRat,
                low_hz=4,
                high_hz=7
            ),
            nn.AvgPool2d(
                kernel_size=(1,poolFB),
                stride=(1,poolFB)
            ),
            # nn.LayerNorm(normalized_shape=samples//4),
            nn.BatchNorm2d(num_features=F1),
            nn.CELU(),
            nn.Dropout(p=droRate_FB)
        )

        self.filterBank_α = nn.Sequential(
            SincConv2d(
                in_channels=1, 
                out_channels=F1,
                kernel_size=L,
                sample_rate=samRat,
                low_hz=8,
                high_hz=13
            ),
            nn.AvgPool2d(
                kernel_size=(1,poolFB),
                stride=(1,poolFB)
            ),
            # nn.LayerNorm(normalized_shape=samples//4),
            nn.BatchNorm2d(num_features=F1),
            nn.CELU(),
            nn.Dropout(p=droRate_FB)
        )

        self.filterBank_β = nn.Sequential(
            SincConv2d(
                in_channels=1, 
                out_channels=F1,
                kernel_size=L,
                sample_rate=samRat,
                low_hz=14,
                high_hz=30
            ),
            nn.AvgPool2d(
                kernel_size=(1,poolFB),
                stride=(1,poolFB)
            ),
            # nn.LayerNorm(normalized_shape=samples//4),
            nn.BatchNorm2d(num_features=F1),
            nn.CELU(),
            nn.Dropout(p=droRate_FB)
        )

        self.filterBank_γ = nn.Sequential(
            SincConv2d(
                in_channels=1, 
                out_channels=F1,
                kernel_size=L,
                sample_rate=samRat,
                low_hz=31,
                high_hz=40
            ),
            nn.AvgPool2d(
                kernel_size=(1,poolFB),
                stride=(1,poolFB)
            ),
            # nn.LayerNorm(normalized_shape=samples//4),
            nn.BatchNorm2d(num_features=F1),
            nn.CELU(),
            nn.Dropout(p=droRate_FB)
        )

        self.spatialFeatExt = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=F1,
                out_channels=F1*D,
                kernel_size=(channels,1),
                groups=F1,
                bias=False,
                max_norm=.5
            ),
            nn.AvgPool2d(
                kernel_size=(1,4),
                stride=(1,4)
            ),
            nn.Dropout(p=dropoutRate)
        )

        self.temporalFeatExt = nn.Sequential(
            nn.Conv2d(
                in_channels=F1*D,
                out_channels=F1*D,
                kernel_size=(1,17),
                padding='same',
                groups=F1*D,
                bias=False
            ),
            # nn.LayerNorm(normalized_shape=samples//16),
            nn.BatchNorm2d(num_features=F1*D),
            nn.CELU(),
            nn.Dropout(p=dropoutRate)
        )

        self.featSelection = nn.Sequential(
            CSA(),
            nn.AvgPool2d(
                kernel_size=(1,4),
                stride=(1,4)
            ),
            # nn.LayerNorm(normalized_shape=samples//64),
            nn.BatchNorm2d(num_features=F1*D*4),
            nn.CELU(),
            nn.Dropout(p=dropoutRate)
        )

        self.classification = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                in_features=4*F1*D*15,
                out_features=n_classes,
                max_norm=.2
            ),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)

        block_θ = self.filterBank_θ(x)
        block_θ = self.spatialFeatExt(block_θ)
        block_θ = self.temporalFeatExt(block_θ)

        block_α = self.filterBank_α(x)
        block_α = self.spatialFeatExt(block_α)
        block_α = self.temporalFeatExt(block_α)

        block_β = self.filterBank_β(x)
        block_β = self.spatialFeatExt(block_β)
        block_β = self.temporalFeatExt(block_β)

        block_γ = self.filterBank_γ(x)
        block_γ = self.spatialFeatExt(block_γ)
        block_γ = self.temporalFeatExt(block_γ)

        x = torch.cat((block_θ,block_α,block_β,block_γ), dim=1)

        x = self.featSelection(x)
        out = self.classification(x)

        return out


#%%
###============================ Initialization parameters ============================###
channels = 22
samples = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = FB_Sinc_CSANet()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()