'''SincConv2d from Mirco Ravanelli et al 2018
Sinc-based convolution

    Notes
    -----
    This implementation has a slight modification from the original code
    and it is taken from the code by Ingolfsson et al at https://github.com/mravanelli/SincNet/tree/master

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math



#%%
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)
    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])
    return y

#%%
class SincConv2d(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
        in_channels : `int`, Number of input channels. Must be 1.
        out_channels : `int`, Number of filters.
        kernel_size : `int`, Filter length.
        sample_rate : `int`, optional Sample rate. Defaults to 200.
        low_hz : `int`, Initialize the low cutoff frequency. Defaults to 4.
        high_hz : `int`, Initialize the high cutoff frequency. Defaults to 7.
    Usage
    -----
        See `torch.nn.Conv1d`
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, in_channels, out_channels, kernel_size, sample_rate=200, stride=1, padding='same', 
                 dilation=1, bias=False, groups=1, min_low_hz=1, min_band_hz=2, low_hz=None, high_hz=None):
        super(SincConv2d,self).__init__()

        if in_channels%groups != 0:
            raise ValueError('in_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.stride = (1,stride)
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        if low_hz is None:
            low_hz = self.min_low_hz
        if high_hz is None:
            high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        elif high_hz > self.sample_rate / 2:
            msg = "The highest cutoff frequency must be less than the Nyquist frequency (here, high_hz = {%i})" % (high_hz)
            raise ValueError(msg)
        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low+self.min_band_hz+torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band=(high-low)[:,0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])

        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        band_pass = band_pass / (2*band[:,None])
        
        self.filters = (band_pass).view(self.out_channels, self.in_channels//self.groups, 1, self.kernel_size)

        return F.conv2d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=self.groups)
    

#%%
###============================ Initialization parameters ============================###
channels = 22
samples = 128

###============================ main function ============================###
def main():
    input = torch.randn(32, 1, channels, samples) # .to(device)
    sincConv2d = SincConv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=5,
    ) # .to(device)
    out = sincConv2d(input)
    print('===============================================================')
    print('out', out.shape)

if __name__ == "__main__":
    main()