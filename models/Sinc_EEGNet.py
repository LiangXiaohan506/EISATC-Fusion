'''Sinc_EEGNet from Alessandro Bria et al 2021.
See details at https://doi.org/10.1007/978-3-030-68763-2_40

    Notes
    -----
    Some modifications have been made to the network structure and initialization parameters

    References
    ----------
    Bria, A., Marrocco, C., Tortorella, F. (2021). 
    Sinc-Based Convolutional Neural Networks for EEG-BCI-Based Motor Imagery Classification. 
    In: , et al. Pattern Recognition. ICPR International Workshops and Challenges. 
    ICPR 2021. Lecture Notes in Computer Science(), vol 12661. Springer, Cham. 
    https://doi.org/10.1007/978-3-030-68763-2_40
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
class Sinc_EEGNet(nn.Module):
    def __init__(self, F1=32, L=64, D=2, samRat=250, samples=1000, channels=22, n_classes=4):
        super(Sinc_EEGNet, self).__init__()
        F2 = F1*D

        self.block_1 = nn.Sequential(
            SincConv2d(
                in_channels=1,
                out_channels=F1, 
                kernel_size=L,
                sample_rate=samRat
            ),
            # nn.AvgPool2d(
            #     kernel_size=(1,4),
            #     stride=(1,4)
            # ),
            # nn.LayerNorm(normalized_shape=samples//4),
            nn.BatchNorm2d(num_features=F1)
            # nn.CELU(),
            # nn.Dropout(p=0.5)
        )

        self.block_2 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=F1,
                out_channels=F1*D,
                kernel_size=(channels,1),
                groups=F1,
                bias=False,
                max_norm=1.
            ),
            nn.AvgPool2d(
                kernel_size=(1,4),
                stride=(1,4)
            ),
            # nn.LayerNorm(normalized_shape=samples//16),
            nn.BatchNorm2d(num_features=F1*D),
            nn.CELU(),
            nn.Dropout(p=0.5)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=F2,
                out_channels=F2,
                kernel_size=(1,16),
                padding='same',
                groups=F2,
                bias=False
            ),
            # nn.LayerNorm(normalized_shape=samples//16),
            # nn.BatchNorm2d(num_features=F1*D),
            # nn.Dropout(p=0.5),
            nn.Conv2d(
                in_channels=F2,
                out_channels=F2,
                kernel_size=1,
                bias=False
            ),
            nn.AvgPool2d(
                kernel_size=(1,8),
                stride=(1,8)
            ),
            # nn.LayerNorm(normalized_shape=samples//64),
            nn.BatchNorm2d(num_features=F2),
            nn.CELU(),
            nn.Dropout(p=0.5)
        )

        self.block_4 = nn.Sequential(
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
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        out = self.block_4(x)
        return out


#%%
###============================ Initialization parameters ============================###
channels = 22
samples = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = Sinc_EEGNet()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()