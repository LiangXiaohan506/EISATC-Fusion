""" Attention Temporal Convolutional Network (ATCNet) from Saudi Arabia et al 2022.
See details at https://ieeexplore.ieee.org/document/9852687/

The original code for this model is available at https://github.com/Altaheri/EEG-ATCNet/tree/main

    References
    ----------
    title={Physics-informed attention temporal convolutional network for EEG-based motor imagery classification}, 
    author={Altaheri, Hamdi and Muhammad, Ghulam and Alsulaiman, Mansour},
    journal={IEEE Transactions on Industrial Informatics}, 
    year={2022},
    doi={10.1109/TII.2022.3197419}
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
from utils.MHSA_util import MultiHeadSelfAttention
from utils.TCN_util import TemporalConvNet
from utils.util import Conv2dWithConstraint,LinearWithConstraint



#%%
class ATCNet(nn.Module):
    def __init__(self, eeg_chans=22, eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_filters=32, tcn_kernelSize=4, tcn_dropout=0.3, n_windows=5, n_classes=4, device='cpu'):
        super(ATCNet, self).__init__()
        self.n_windows = n_windows
        self.n_classes = n_classes
        self.device = device

        self.conv_block = Conv_block(
            F1          =eegn_F1,
            D           =eegn_D,
            eeg_chans   =eeg_chans,
            kernLength  =eegn_kernelSize,
            poolSize    =eegn_poolSize,
            dropout     =eegn_dropout
        )

        self.ln = nn.LayerNorm(
            normalized_shape=eegn_F1*eegn_D,
            eps=1e-6)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=eegn_F1*eegn_D,
            num_heads=2,
            dropout=0.5,
            batch_first=True
        )

        self.tcn_block = TemporalConvNet(
            num_inputs  =eegn_F1*eegn_D,
            num_channels=[tcn_filters, tcn_filters],
            kernel_size =tcn_kernelSize,
            dropout     =tcn_dropout
        )

        self.fc1 = LinearWithConstraint(
            in_features=eegn_F1*eegn_D,
            out_features=n_classes,
            max_norm=.25
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        x = self.conv_block(x)
        x = torch.squeeze(x, dim=2) # (batch, F1*D, Tc)

        sw_concat = torch.zeros(self.n_windows, x.shape[0], self.n_classes).to(self.device)
        for i in range(self.n_windows):
            st=i
            end=x.shape[2]-self.n_windows+i+1
            x_ = x[:, :, st:end] # (batch, F1*D, Tc)

            x_ = torch.transpose(x_, len(x_.shape)-2, len(x_.shape)-1) # (batch, Tc, F1*D)
            x_ = self.ln(x_)
            x_, _ = self.multihead_attn(x_, x_, x_)
            x_ = torch.transpose(x_, len(x_.shape)-2, len(x_.shape)-1) # (batch, F1*D, Tc)
            x_ = self.tcn_block(x_) # (batch, F1*D, Tc)
            x_ = x_[:, :, -1] # (batch, F1*D)

            sw_concat[i, :, :] = self.fc1(x_)

        sw_concat = sw_concat[:]
        if len(sw_concat) > 1: # more than one window
            sw_concat = torch.mean(sw_concat[:], dim=0) # (batch, n_classes)
        else: # one window (# windows = 1)
            sw_concat = sw_concat[0]

        out = self.softmax(sw_concat)

        return out


#%%
class Conv_block(nn.Module):
    def __init__(self, F1=4, kernLength=64, poolSize=8, D=2, eeg_chans=22, dropout=0.1):
        super(Conv_block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=F1,
            kernel_size=(1,kernLength),
            padding='same',
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=F1)
        
        self.conv2 = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1*D,
            kernel_size=(eeg_chans,1),
            groups=F1,
            bias=False,
            max_norm=1.
        )
        self.bn2 = nn.BatchNorm2d(num_features=F1*D)
        self.act2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d(
            kernel_size=(1,8),
            stride=(1,8)
        )
        self.drop2 = nn.Dropout(p=dropout)

        self.conv3 = nn.Conv2d(
            in_channels=F1*D,
            out_channels=F1*D,
            kernel_size=(1,16),
            padding='same',
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=F1*D)
        self.act3 = nn.ELU()
        self.avgpool3 = nn.AvgPool2d(
            kernel_size=(1, poolSize),
            stride=(1, poolSize)
        )
        self.drop3 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.drop2(self.avgpool2(self.act2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.avgpool3(self.act3(self.bn3(self.conv3(x)))))
        return x


#%%
###============================ Initialization parameters ============================###
channels = 22
samples  = 1000
device   = torch.device('cpu')

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = ATCNet(eeg_chans=22)
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()