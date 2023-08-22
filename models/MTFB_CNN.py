""" MTFB-CNN model from Hongli Li et al 2023.
See details at https://doi.org/10.1016/j.bspc.2022.104066

    Notes
    -----
    The initial values in this model are based on the values identified by the authors

    References
    ----------
    Li H, Chen H, Jia Z, et al. 
    A parallel multi-scale time-frequency block convolutional neural network based on channel attention module for motor imagery classification[J]. 
    Biomedical Signal Processing and Control, 2023, 79: 104066.
"""

import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from torchstat import stat
import yaml



class TFB(nn.Module):
    def __init__(self, kerSize_1, kerSize_2, kerSize_3, kerStr, out_chan, pool_ker, pool_str):
        super(TFB, self).__init__()

        self.kerSize_1 = kerSize_1

        self.path_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_chan,
                kernel_size=(1,kerSize_1),
                stride=(1,kerStr),
                padding=(0, round(kerSize_1/2)-1 if kerSize_1%2==0 else round(kerSize_1/2)-2)
            ),
            nn.BatchNorm2d(num_features=out_chan),
            nn.SELU()
        )

        self.path_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_chan,
                kernel_size=(1,kerSize_2),
                stride=(1,kerStr),
                padding=(0,round(kerSize_2/2)-1)
            ),
            nn.BatchNorm2d(num_features=out_chan),
            nn.SELU()
        )

        self.path_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_chan,
                kernel_size=(1,kerSize_3),
                stride=(1,kerStr),
                padding=(0,round(kerSize_3/2)-1)
            ),
            nn.BatchNorm2d(num_features=out_chan),
            nn.SELU()
        )

        self.path_4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=(1,pool_ker),
                stride=(1,pool_str),
                padding=(0,round(pool_ker/2)-1)
            ),
            nn.Conv2d(
                in_channels=1,
                out_channels=out_chan,
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(num_features=out_chan),
            nn.SELU()
        )
    
    def forward(self, x):
        p1 = self.path_1(x)
        p2 = self.path_2(x)
        p3 = self.path_3(x)
        p4 = self.path_4(x)
        out = torch.cat((p1,p2,p3,p4), dim=1)
        return out


class ResneXt(nn.Module):
    def __init__(self, in_chan, kerSize, out_chan, dropoutRate):
        super(ResneXt, self).__init__()

        self.path_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=(1,kerSize),
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(num_features=out_chan),
            nn.ELU(),
            nn.Dropout(p=dropoutRate),
            nn.Conv2d(
                in_channels=out_chan,
                out_channels=out_chan,
                kernel_size=(1,kerSize),
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(num_features=out_chan)
        )

        self.path_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=(1,kerSize),
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(num_features=out_chan),
            nn.ELU(),
            nn.Dropout(p=dropoutRate),
            nn.Conv2d(
                in_channels=out_chan,
                out_channels=out_chan,
                kernel_size=(1,kerSize),
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(num_features=out_chan)
        )

        self.path_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=(1,kerSize),
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(num_features=out_chan),
            nn.ELU(),
            nn.Dropout(p=dropoutRate),
            nn.Conv2d(
                in_channels=out_chan,
                out_channels=out_chan,
                kernel_size=(1,kerSize),
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(num_features=out_chan)
        )

        self.path_4 = nn.Conv2d(
                in_channels=in_chan,
                out_channels=in_chan,
                kernel_size=1,
                stride=1
            )
        
        self.activate = nn.ELU()

    def forward(self, x):
        p1 = self.path_1(x)
        p2 = self.path_2(x)
        p3 = self.path_3(x)
        p123 = torch.cat((p1,p2,p3), dim=1)
        p4 = self.path_4(x)
        x = p123+p4
        out = self.activate(x)
        return out


class CAM(nn.Module):
    def __init__(self, chanSize, reduRatio):
        super(CAM, self).__init__()

        self.maxPool = nn.AdaptiveMaxPool2d(output_size=(chanSize, 1))
        self.maxFc_1 = nn.Linear(
            in_features=chanSize,
            out_features=chanSize//reduRatio)
        self.maxFc_2 = nn.Linear(
            in_features=chanSize//reduRatio,
            out_features=chanSize)

        self.avgPool = nn.AdaptiveAvgPool2d(output_size=(chanSize, 1))
        self.avgFc_1 = nn.Linear(
            in_features=chanSize,
            out_features=chanSize//reduRatio)
        self.avgFc_2 = nn.Linear(
            in_features=chanSize//reduRatio,
            out_features=chanSize)
        
        self.activate = nn.Sigmoid()

    def forward(self, x):
        res = x

        max_x = self.maxPool(x)
        max_x = torch.squeeze(max_x, dim=-1)
        max_x = self.maxFc_1(max_x)
        max_x = self.maxFc_2(max_x)

        avg_x = self.avgPool(x)
        avg_x = torch.squeeze(avg_x, dim=-1)
        avg_x = self.avgFc_1(avg_x)
        avg_x = self.avgFc_2(avg_x)

        x = max_x+avg_x
        x = self.activate(x)
        x = torch.unsqueeze(x, dim=3)
        x = x*res
        out = x+res
        return out


class MTFB_CNN(nn.Module):
    def __init__(self, chanSize=22, n_classes=4):
        super(MTFB_CNN, self).__init__()

        self.branch_a = nn.Sequential(
            TFB(
                kerSize_1=4,
                kerSize_2=6,
                kerSize_3=10,
                kerStr=2,
                out_chan=6,
                pool_ker=3,
                pool_str=2
            ),
            nn.MaxPool2d(
                kernel_size=(1,4),
                stride=(1,4)
            ),
            nn.BatchNorm2d(num_features=24),
            nn.Dropout(p=0.1),
            ResneXt(
                in_chan=24,
                kerSize=8,
                out_chan=8,
                dropoutRate=0.1
            ),
            nn.Dropout(p=0.1),
            CAM(
                chanSize=chanSize,
                reduRatio=2
            ),
            nn.MaxPool2d(
                kernel_size=(1,6),
                stride=(1,6)
            ),
            nn.Flatten()
        )

        self.branch_b = nn.Sequential(
            TFB(
                kerSize_1=15,
                kerSize_2=30,
                kerSize_3=45,
                kerStr=3,
                out_chan=6,
                pool_ker=10,
                pool_str=3
            ),
            nn.MaxPool2d(
                kernel_size=(1,3),
                stride=(1,3)
            ),
            nn.BatchNorm2d(num_features=24),
            nn.Dropout(p=0.1),
            ResneXt(
                in_chan=24,
                kerSize=5,
                out_chan=8,
                dropoutRate=0.1
            ),
            nn.Dropout(p=0.1),
            CAM(
                chanSize=chanSize,
                reduRatio=2
            ),
            nn.MaxPool2d(
                kernel_size=(1,4),
                stride=(1,4)
            ),
            nn.Flatten()
        )

        self.branch_c = nn.Sequential(
            TFB(
                kerSize_1=50,
                kerSize_2=70,
                kerSize_3=120,
                kerStr=4,
                out_chan=6,
                pool_ker=20,
                pool_str=4
            ),
            nn.MaxPool2d(
                kernel_size=(1,3),
                stride=(1,3)
            ),
            nn.BatchNorm2d(num_features=24),
            nn.Dropout(p=0.1),
            ResneXt(
                in_chan=24,
                kerSize=5,
                out_chan=8,
                dropoutRate=0.1
            ),
            nn.Dropout(p=0.1),
            CAM(
                chanSize=chanSize,
                reduRatio=2
            ),
            nn.MaxPool2d(
                kernel_size=(1,4),
                stride=(1,4)
            ),
            nn.Flatten()
        )

        self.fc = nn.Linear(
            in_features=(20+27+20)*chanSize*24,
            out_features=n_classes
        )
        self.activate = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # input shape (batch_size, C, T)
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        # input shape (batch_size, 1, C, T)
        bra_a = self.branch_a(x) # (batch, out_chan, channels, 20)
        bra_b = self.branch_b(x) # (batch, out_chan, channels, 27)
        bra_c = self.branch_c(x) # (batch, out_chan, channels, 20)
        x = torch.cat((bra_a, bra_b, bra_c), dim=-1)
        x = self.fc(x)
        out = self.activate(x)
        return out


###============================ Initialization parameters ============================###
channels = 22
samples = 1000
n_classes = 4

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = MTFB_CNN(channels, n_classes)
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()