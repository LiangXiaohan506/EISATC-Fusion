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

import math
import torch
from torch import nn
from einops import rearrange
from util import Conv2dWithConstraint



#%%
def relative_pos_dis(height=1, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    if height > 1:
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2 # Wh*Ww, Wh*Ww
    else:
        relative_coords = coords_flatten[1, :, None] - coords_flatten[1, None, :]  # Wh*Ww, Wh*Ww
        relative_coords = relative_coords.contiguous()  # Wh*Ww, Wh*Ww
        dis = relative_coords # Wh*Ww, Wh*Ww
    return  dis



#%%
class CNNAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, keral_size=3, dropout=0., patch_height=1, patch_width=1, max_norm1=1., max_norm2=1., device='cpu', groups=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.patch_width = patch_width

        if groups:
            self.to_qkv = Conv2dWithConstraint(dim, inner_dim*3, kernel_size=(1, keral_size), padding='same', bias=False, max_norm=max_norm1, groups=dim)
        else:
            self.to_qkv = Conv2dWithConstraint(dim, inner_dim*3, kernel_size=(1,keral_size), padding='same', bias=False, max_norm=max_norm1)

        self.dis = relative_pos_dis(patch_height, patch_width).to(device) # n n
        self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.sig = nn.Sigmoid()
        self.ones_matrix = torch.ones(patch_height*patch_width, patch_height*patch_width).to(device)

        self.to_out = nn.Sequential(
            Conv2dWithConstraint(inner_dim, dim, kernel_size=1, padding=0, bias=False, max_norm=max_norm2),
            nn.BatchNorm2d(dim), # inner_dim
            nn.ELU(), # inplace=True
            nn.Dropout(p=dropout)
        )

    def forward(self, x, mode="train", smooth=1e-4):
        qkv = self.to_qkv(x).chunk(3, dim=1) # b (g d) h w
        q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.heads), qkv) # b g n d
        attn = torch.matmul(q, k.transpose(-1, -2)) # b g n n
        qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1)+smooth)[:, :, :, None] * torch.sqrt(torch.sum(k ** 2, dim=-1)+smooth)[:, :, None, :] + smooth
        attn = attn/qk_norm # b g n n

        f = self.ones_matrix*(self.sig(self.headsita)*(self.patch_width-1)+1)[:, None, None]
        cycle_attn = 0.5*torch.cos(f*self.dis)+0.5 # g n n
        attention = attn * cycle_attn[None, :, :, :] # b g n n

        out = torch.matmul(attention, v) # b g n d

        out = rearrange(out, 'b g (h w) d -> b (g d) h w', h=x.shape[2])
        if mode=="train":
            return self.to_out(out)
        elif mode=="test":
            return self.to_out(out), attention, attn, cycle_attn
