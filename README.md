# EISATC-Fusion

This repository provides code for the Attention Temporal Convolutional Network [(EISATC-Fusion)](https://ieeexplore.ieee.org/abstract/document/10480732) proposed in the paper: [EISATC-Fusion: Inception Self-Attention Temporal Convolutional Network Fusion for Motor Imagery EEG Decoding](https://ieeexplore.ieee.org/abstract/document/10480732)

Authors: Guangjin Liang, Dianguo Cao, Jinqiang Wang, Zhongcai Zhang, Yuqiang Wu

Collaborative Innovation Centre for Rehabilitation and Care Robotics, College of Engineering, Qufu Normal University, Rizhao, China


EISATC-Fusion model consists of four modules: the EEGNet DS Inception (EDSI) module, cnnCos multi-head self-attention (cnnCosMSA) module, temporal depthwise separable convolutional network (TDSCN) module, and fusion module.



<p align="center">
<img src="pictures/Overall architecture.png" alt="The overall architecture of EISATC-Fusion" width="400"/>
</p>
<p align="center">
The overall architecture of EISATC-Fusion.
</p>

1. The EDSI is inspired by the EEGNet architecture introduced in [14], and the layer of SeparableConv2D in EEGNet is replaced by the DS Inception block. The EDSI uses normal convolution and depthwise (DW) convolution to extract the temporal and spatial features and uses a depthwise separable (DS) inception block to extract the multi-scale time features.


<p align="center">
<img src="pictures/EEGNet DS Inception (EDSI) Module.png" alt="Architecture of the EDSI" width="800"/>
</p>
<p align="center">
Architecture of the EDSI.
</p>

<p align="center">
<img src="pictures/Inception DW Conv.png" alt="Architecture of the DS Inception (DSI) block consisting of four paths" width="400"/>
</p>
<p align="center">
Architecture of the DS Inception (DSI) block consisting of four paths.
</p>

2. The cnnCos Multi-head Self-Attention module is designed based on the CNN to address the issue of attention collapse during the processing of EEG signals by MSA. Cos attention is added to enhance the attention value and improve the interpretability of the model.

<p align="center">
<img src="pictures/cnnCos Multi-head Self-Attention (cnnCosMSA) Module.png" alt="Architecture of the cnnCosMSA" width="800"/>
</p>
<p align="center">
Architecture of the cnnCosMSA.
</p>

3. In order to reduce the number of parameters of the TCN module without reducing the decoding performance of the module, we improved TCN by replacing dilated causal convolution with dilated causal DS convolution, which contains a layer of the dilated causal depthwise convolution and a layer of pointwise convolution.

<p align="center">
<img src="pictures/Temporal Depthwise Separable Convolutional Network (TDSCN) Module.png" alt="Architecture of the TDSCN" width="800"/>
</p>
<p align="center">
Architecture of the TDSCN
</p>

4. The fusion module consists of two parts: feature fusion and decision fusion.






## Dataset 
The [BCI Competition IV-2a](https://www.bbci.de/competition/iv/#dataset2a) and  [BCI Competition IV-2b](https://www.bbci.de/competition/iv/#dataset2b) dataset needs to be downloaded and the data path placed at 'data_path' variable in [*main.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main.py) file. The dataset can be downloaded from [here](http://bnci-horizon-2020.eu/database/data-sets).

## Development environment
Models were trained and tested by a single GPU, Nvidia [GTX 3090 24GB](https://www.nvidia.com/en-me/geforce/graphics-cards/30-series/) (Driver Version: [510.108.03](https://www.nvidia.com/download/driverResults.aspx/188599/en-us/), [CUDA 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive)), using Python 3.7 with [PyTorch](https://pytorch.org/) framework. [Anaconda 3](https://www.anaconda.com/products/distribution) was used on [Ubuntu 18.04.6 LTS](https://releases.ubuntu.com/bionic/).
The following packages are required:
* PyTorch   : 1.12.1
* TorchVison: 0.13.1
* matplotlib 3.5
* NumPy 1.20
* scikit-learn 1.0
* SciPy 1.7

## References
If you find this work useful in your research, please use the following BibTeX entry for citation

```
@ARTICLE{10480732,
  author={Liang, Guangjin and Cao, Dianguo and Wang, Jinqiang and Zhang, Zhongcai and Wu, Yuqiang},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={EISATC-Fusion: Inception Self-Attention Temporal Convolutional Network Fusion for Motor Imagery EEG Decoding}, 
  year={2024},
  volume={32},
  number={},
  pages={1535-1545},
  keywords={Feature extraction;Brain modeling;Electroencephalography;Convolution;Convolutional neural networks;Decoding;Kernel;Brain–computer interface (BCI);motor imagery (MI);attention collapse;temporal convolution network (TCN);transfer learning},
  doi={10.1109/TNSRE.2024.3382226}}

```
