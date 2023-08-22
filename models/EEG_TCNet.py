""" EEGTCNet model from Ingolfsson et al 2020.
See details at https://arxiv.org/abs/2006.00622

The original code for this model is available at https://github.com/iis-eth-zurich/eeg-tcnet

    Notes
    -----
    The initial values in this model are based on the values identified by the authors
    
    References
    ----------
    .. Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N.,
        Cavigelli, L., & Benini, L. (2020, October). 
        Eeg-tcnet: An accurate temporal convolutional network
        for embedded motor-imagery brain-machine interfaces. 
        In 2020 IEEE International Conference on Systems, 
        Man, and Cybernetics (SMC) (pp. 2958-2965). IEEE.
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
from utils.EEGNet_util import EEGNet_util
from utils.TCN_util import TemporalConvNet
from utils.util import LinearWithConstraint



#%%
class EEG_TCNet(nn.Module):
    def __init__(self, eeg_chans=22, F1=8, D=2, EEGkerSize=32, EEGkerSize_Tem=16, EEG_dropout=0.3, 
                 tcn_filters=12, tcn_kernelSize=4, tcn_dropout=0.3, n_classes=4):
        super(EEG_TCNet, self).__init__()
        F2 = F1*D

        self.eegNet = EEGNet_util(
            eeg_chans   =eeg_chans, 
            dropoutRate =EEG_dropout,
            kerSize     =EEGkerSize,
            kerSize_Tem =EEGkerSize_Tem,
            F1          =F1,
            D           =D
        )

        self.tcn_block = TemporalConvNet(
            num_inputs  =F2,
            num_channels=[tcn_filters, tcn_filters],
            kernel_size =tcn_kernelSize,
            dropout     =tcn_dropout
        )

        self.flatten = nn.Flatten()
        self.dense = LinearWithConstraint(
            in_features=tcn_filters,
            out_features=n_classes,
            max_norm=.25
        )
        self.dense.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)

        x = self.eegNet(x) # NCHW
        x = torch.squeeze(x, dim=2) # NCW

        x = self.tcn_block(x) # NWC
        x = x[:, :, -1] # NC

        x = self.flatten(x)
        x = self.dense(x)
        out = self.softmax(x)

        return out
    

#%%
###============================ Initialization parameters ============================###
channels = 22
samples = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = EEG_TCNet()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()