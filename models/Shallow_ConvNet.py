""" Keras implementation of the Shallow Convolutional Network as described
in Schirrmeister et. al. (2017), Human Brain Mapping.
See details at https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730

The original code for this model is available at:
    https://github.com/braindecode/braindecode

Note that the default parameters of the model come from 
    https://github.com/Altaheri/EEG-ATCNet/tree/main
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
class Shallow_ConvNet(nn.Module):
    def __init__(self, conv_channel_temp=40, kernel_size_temp=25, conv_channel_spat=40, kernel_size_spat=22,
                 pooling_size=75, pool_stride_size=15, dropoutRate=0.5, n_classes=4, class_kernel_size=61, bias=False) :
        super(Shallow_ConvNet, self).__init__()

        self.temp_conv = Conv2dWithConstraint(
            in_channels=1,
            out_channels=conv_channel_temp,
            kernel_size=(1, kernel_size_temp),
            stride=1,
            bias=bias,
            max_norm=1.
        )

        self.spat_conv = Conv2dWithConstraint(
            in_channels=conv_channel_temp,
            out_channels=conv_channel_spat,
            kernel_size=(kernel_size_spat, 1),
            stride=1,
            bias=bias,
            max_norm=1.
        )

        self.bn = nn.BatchNorm2d(
            num_features=conv_channel_spat,
            momentum=0.9
        )

        # slef.act_conv = x*x

        self.pooling = nn.AvgPool2d(
            kernel_size=(1, pooling_size),
            stride=(1, pool_stride_size))

        # slef.act_pool = log(max(x, eps))

        self.dropout = nn.Dropout(p=dropoutRate)

        # self.class_conv = nn.Conv2d(
        #     in_channels=conv_channel_spat,
        #     out_channels=n_classes,
        #     kernel_size=(1, class_kernel_size),
        #     bias=bias)

        self.flatten = nn.Flatten()
        self.dense = LinearWithConstraint(
            in_features=2440,
            out_features=n_classes,
            max_norm=.5
        )

        self.softmax = nn.Softmax(dim=1)

    def safe_log(self, x):
        """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
        return torch.log(torch.clamp(x, min=1e-6))
    
    def forward(self, x):
        # input shape (batch_size, C, T)
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        # input shape (batch_size, 1, C, T)
        x = self.temp_conv(x)
        x = self.spat_conv(x)
        x = self.bn(x)
        x = x*x # conv_activate
        x = self.pooling(x)
        x = self.safe_log(x) # pool_activate
        x = self.dropout(x)
        # x = self.class_conv(x)
        # x = torch.squeeze(x, -1)
        # x = torch.squeeze(x, -1)
        x = self.flatten(x)
        x = self.dense(x)
        out= self.softmax(x)

        return out


#%%
###============================ Initialization parameters ============================###
channels = 22
samples = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = Shallow_ConvNet()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()
