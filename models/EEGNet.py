""" EEGNet model from Lawhern et al 2018
See details at https://arxiv.org/abs/1611.08024

The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels

    Notes
    -----
    The initial values in this model are based on the values identified by the authors
    
    References
    ----------
    .. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
        S. M., Hung, C. P., & Lance, B. J. (2018).
        EEGNet: A Compact Convolutional Network for EEG-based
        Brain-Computer Interfaces.
        arXiv preprint arXiv:1611.08024.
"""

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
from utils.util import Conv2dWithConstraint, LinearWithConstraint



#%%
class EEGNet(nn.Module):
    def __init__(self, eeg_chans=22, samples=1000, dropoutRate=0.5, kerSize=64, kerSize_Tem=16, F1=8, D=2, bias=False, n_classes=4):
        super(EEGNet, self).__init__()
        F2 = F1*D

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=F1, 
                kernel_size=(1, kerSize), 
                stride=1,
                padding='same',
                bias=bias
            ), 
            nn.BatchNorm2d(num_features=F1) 
        )

        self.depthwiseConv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=F1, 
                out_channels=F1*D,
                kernel_size=(eeg_chans, 1),
                groups=F1,
                bias=bias, 
                max_norm=1.
            ), 
            nn.BatchNorm2d(num_features=F1*D),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,4),
                stride=(1,4)
            ),
            nn.Dropout(p=dropoutRate)
        )

        self.seqarableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=F2, 
                out_channels=F2,
                kernel_size=(1,kerSize_Tem),
                stride=1,
                padding='same',
                groups=F2,
                bias=bias
            ),
            nn.Conv2d(
                in_channels=F2,
                out_channels=F2,
                kernel_size=(1,1),
                stride=1,
                bias=bias
            ),
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,8),
                stride=(1,8)
            ),
            nn.Dropout(p=dropoutRate)
        )

        self.class_head = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                in_features=samples//32*F2,
                out_features=n_classes,
                max_norm=.25
            ),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        x = self.block1(x)
        x = self.depthwiseConv(x)
        x = self.seqarableConv(x)
        x = self.class_head(x)
        return x


#%%
###============================ Initialization parameters ============================###
channels = 22
samples  = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = EEGNet()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()