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
class Deep_ConvNet(nn.Module):
    def __init__(self, conv_channel_temp=25, kernel_size_temp=10, conv_channel_spat=25, kernel_size_spat=22, pooling_size_1=3, pool_stride_size_1=3,
                 dropoutRate_2=0.5, conv2_channel_out=50, conv2_kernel_size=10, pooling_size_2=3, pool_stride_size_2=3,
                 dropoutRate_3=0.5, conv3_channel_out=100, conv3_kernel_size=10, pooling_size_3=3, pool_stride_size_3=3,
                 dropoutRate_4=0.5, conv4_channel_out=200, conv4_kernel_size=10, pooling_size_4=3, pool_stride_size_4=3,
                 n_classes=4, class_kernel_size=2, bias=False) :
        super(Deep_ConvNet, self).__init__()

        self.block1 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=conv_channel_temp,
                kernel_size=(1, kernel_size_temp),
                stride=1,
                bias=bias,
                max_norm=1.
            ),
            Conv2dWithConstraint(
                in_channels=conv_channel_temp,
                out_channels=conv_channel_spat,
                kernel_size=(kernel_size_spat, 1),
                stride=1,
                bias=bias,
                max_norm=1.
            ),
            nn.BatchNorm2d(num_features=conv_channel_spat),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, pooling_size_1),
                stride=(1, pool_stride_size_1))
        )

        self.block2 = nn.Sequential(
            nn.Dropout(p=dropoutRate_2),
            Conv2dWithConstraint(
                in_channels=conv_channel_spat,
                out_channels=conv2_channel_out,
                kernel_size=(1, conv2_kernel_size),
                stride=1,
                bias=bias,
                max_norm=1.
            ),
            nn.BatchNorm2d(num_features=conv2_channel_out),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, pooling_size_2),
                stride=(1, pool_stride_size_2))
        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=dropoutRate_3),
            Conv2dWithConstraint(
                in_channels=conv2_channel_out,
                out_channels=conv3_channel_out,
                kernel_size=(1, conv3_kernel_size),
                stride=1,
                bias=bias,
                max_norm=1.
            ),
            nn.BatchNorm2d(num_features=conv3_channel_out),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, pooling_size_3),
                stride=(1, pool_stride_size_3))
        )

        self.block4 = nn.Sequential(
            nn.Dropout(p=dropoutRate_4),
            Conv2dWithConstraint(
                in_channels=conv3_channel_out,
                out_channels=conv4_channel_out,
                kernel_size=(1, conv4_kernel_size),
                stride=1,
                bias=bias,
                max_norm=1.
            ),
            nn.BatchNorm2d(num_features=conv4_channel_out),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, pooling_size_4),
                stride=(1, pool_stride_size_4))
        )

        # self.class_conv = nn.Conv2d(
        #     in_channels=conv4_channel_out,
        #     out_channels=n_classes,
        #     kernel_size=(1, class_kernel_size),
        #     bias=True)

        self.flatten = nn.Flatten()
        self.dense = LinearWithConstraint(
            in_features=1400,
            out_features=n_classes,
            max_norm=.5
        )

        self.softmax = nn.Softmax(dim=1)

    def safe_log(self, x):
        """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
        return torch.log(torch.clamp(x, min=1e-6))
    
    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # x = self.class_conv(x)
        # x = torch.squeeze(x, -1)
        # x = torch.squeeze(x, -1)
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
    model = Deep_ConvNet()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()
