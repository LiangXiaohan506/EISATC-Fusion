""" 
Copyright (C) 2023 Qufu Normal University, Guangjin Liang
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Guangjin Liang
"""

import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
from torch import nn
from util import Conv2dWithConstraint



#%%
class TemporalInception(nn.Module):
    def __init__(self, in_chan=1, kerSize_1=(1,3), kerSize_2=(1,5), kerSize_3=(1,7),
                 kerStr=1, out_chan=4, pool_ker=(1,3), pool_str=1, bias=False, max_norm=1., point_fusion=True):
        '''
        Inception模块的实现代码,

        ''' 
        super(TemporalInception, self).__init__()
        self.point_fusion = point_fusion

        self.conv1 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_1,
            stride=kerStr,
            padding='same',
            groups=out_chan,
            bias=bias,
            max_norm=max_norm
        )

        self.conv2 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_2,
            stride=kerStr,
            padding='same',
            groups=out_chan,
            bias=bias,
            max_norm=max_norm
        )

        self.conv3 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_3,
            stride=kerStr,
            padding='same',
            groups=out_chan,
            bias=bias,
            max_norm=max_norm
        )

        self.pool4 = nn.MaxPool2d(
            kernel_size=pool_ker,
            stride=pool_str,
            padding=(round(pool_ker[0]/2+0.1)-1,round(pool_ker[1]/2+0.1)-1)
        )
        self.conv4 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=1,
            stride=1,
            bias=bias,
            max_norm=max_norm
        )
    
        if point_fusion:
            self.point_conv = Conv2dWithConstraint(
                in_channels =in_chan,
                out_channels=in_chan,
                kernel_size =1,
                stride      =1,
                bias        =False
            )

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(self.pool4(x))
        out = torch.cat((p1,p2,p3,p4), dim=1)
        if self.point_fusion:
            out = self.point_conv(out)
        return out
