"""TCN_block from Bai et al 2018
Temporal Convolutional Network (TCN)

    Notes
    -----
    This implementation has a slight modification from the original code
    and it is taken from the code by Ingolfsson et al at https://github.com/iis-eth-zurich/eeg-tcnet
    See details at https://arxiv.org/abs/2006.00622

    References
    ----------
    .. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
        An empirical evaluation of generic convolutional and recurrent networks
        for sequence modeling.
        arXiv preprint arXiv:1803.01271.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from util import Conv1dWithConstraint



#%%
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

#%%
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bias=False, WeightNorm=False, group=True, max_norm=1.):
        super(TemporalBlock, self).__init__()
        if group:
            if n_inputs >= n_outputs:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            else:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_inputs)
            self.conv1_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu1 = nn.ELU() # inplace=True
        self.dropout1 = nn.Dropout(dropout)

        if group:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            self.conv2_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        if group:
            self.net = nn.Sequential(self.conv1, self.conv1_point, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.conv2_point, self.chomp2, self.bn2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out+res
        out = self.relu(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, bias=False, WeightNorm=False, group=True, max_norm=1.):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                                     dropout=dropout, bias=bias, WeightNorm=WeightNorm, group=group, max_norm=max_norm)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


#%%
###============================ Initialization parameters ============================###
channels = 22
samples = 250

###============================ main function ============================###
def main():
    input = torch.randn(32, 1, samples)
    TCN = TemporalConvNet(
        num_inputs=1,
        num_channels=[2],
        kernel_size=4,
    )
    out = TCN(input)
    print('===============================================================')
    print('out', out.shape)

if __name__ == "__main__":
    main()

