""" Incep-EEGNet model from RIYAD M et al 2020
See details at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7340940/

    Notes
    -----
    The initial values in this model are based on the values identified by the authors
    
    References
    ----------
    .. Riyad M, Khalil M, Adib A. 
       Incep-EEGNet: A ConvNet for Motor Imagery Decoding. 
       Image and Signal Processing. 2020 Jun 5;12119:103–11. 
       doi: 10.1007/978-3-030-51935-3_11. 
       PMCID: PMC7340940.
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



#%%
class Inception(nn.Module):
    def __init__(self, in_chan=1, kerSize_1=(1,7), kerSize_2=(1,9), kerSize_3=(1,1), out_chan=64,):
        '''
        Inception模块的实现代码,

        ''' 
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=(1,1),
                padding='same',
            ),
            nn.Conv2d(
                in_channels=out_chan,
                out_channels=out_chan,
                kernel_size=kerSize_1,
                padding='same'
            )
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=(1,1),
                padding='same',
            ),
            nn.Conv2d(
                in_channels=out_chan,
                out_channels=out_chan,
                kernel_size=kerSize_2,
                padding='same'
            )
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(1,2),
                padding=(0,1),
            ),
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kerSize_3,
                padding='same'
            )
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=(1,1),
                stride=(1,2),
                padding=(0,250),
            )
        )

    
    def forward(self, x):
        p1 = self.branch1(x)
        p2 = self.branch1(x)
        p3 = self.branch1(x)
        p4 = self.branch1(x)
        out = torch.cat((p1,p2,p3,p4), dim=1)
        return out



#%%
class Incep_EEGNet(nn.Module):
    def __init__(self, channels=22, dropoutRate=0.5, kerSize=16, kerSize_Tem=5, F1=64, D=4, n_classes=4):
        super(Incep_EEGNet, self).__init__()
        F2 = F1*D

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=F1, 
                kernel_size=(1, kerSize), 
                stride=1,
                padding='same'
            ), 
            nn.BatchNorm2d(num_features=F1) 
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1, 
                out_channels=F1*D,
                kernel_size=(channels, 1),
                groups=F1
            ), 
            nn.BatchNorm2d(num_features=F1*D),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,2),
                stride=(1,2)
            ),
            nn.Dropout(p=dropoutRate)
        )

        self.inception = nn.Sequential(
            Inception(in_chan=256, out_chan=64),
            nn.BatchNorm2d(num_features=F1*D),
            nn.ELU()
        )
        
        self.seqarableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=F2, 
                out_channels=F2,
                kernel_size=(1,kerSize_Tem),
                stride=1,
                padding='same',
            ),
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )

        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1)),
            nn.Flatten(),
            nn.Linear(in_features=F2, out_features=n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        x = self.block1(x)
        x = self.depthwiseConv(x)
        x = self.inception(x)
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
    model = Incep_EEGNet()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()