""" TCNet_Fusion model from Musallam et al 2021.
See details at https://doi.org/10.1016/j.bspc.2021.102826
    
    References
    ----------
    .. Musallam, Y.K., AlFassam, N.I., Muhammad, G., Amin, S.U., Alsulaiman,
        M., Abdul, W., Altaheri, H., Bencherif, M.A. and Algabri, M., 2021. 
        Electroencephalography-based motor imagery classification
        using temporal convolutional network fusion. 
        Biomedical Signal Processing and Control, 69, p.102826.
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
from utils.util import Conv2dWithConstraint,LinearWithConstraint



#%%
class TCNet_Fusion(nn.Module):
    def __init__(self, Chans=22, Samples=1000, F1=24, D=2, kernLength=32, dropout_eeg=0.3, 
                 tcn_filters=12, tcn_kernelSize=4, tcn_dropout=0.3, n_classes=4):
        super(TCNet_Fusion, self).__init__()
        F2 = F1*D

        self.eegUtil = EEGNet_util(
            eeg_chans   =Chans,
            kerSize     =kernLength,
            kerSize_Tem =16,
            F1          =F1,
            D           =D,
            dropoutRate =dropout_eeg
        )

        self.flatten_eeg = nn.Flatten()

        self.tcn_block = TemporalConvNet(
            num_inputs  =F2,
            num_channels=[tcn_filters, tcn_filters],
            kernel_size =tcn_kernelSize,
            dropout     =tcn_dropout
        )

        self.flatten_eeg_tcn = nn.Flatten()

        self.dense = LinearWithConstraint(
            in_features=1620,
            out_features=n_classes,
            max_norm=.25
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)

        eeg_out = self.eegUtil(x)
        eeg_out = torch.squeeze(eeg_out, dim=2)
        eeg_out_fal = self.flatten_eeg(eeg_out)

        tcn_out = self.tcn_block(eeg_out)
        eeg_tcn_out = torch.cat((eeg_out, tcn_out), dim=1)
        eeg_tcn_out_fal = self.flatten_eeg_tcn(eeg_tcn_out)

        eeg_tcn_cat = torch.cat((eeg_out_fal, eeg_tcn_out_fal), dim=-1)

        eeg_tcn = self.dense(eeg_tcn_cat)
        out = self.softmax(eeg_tcn)

        return out


#%%
###============================ Initialization parameters ============================###
channels = 22
samples = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = TCNet_Fusion()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()