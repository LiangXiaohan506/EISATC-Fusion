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
import mne
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import cheb2ord
from mne.decoding import CSP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay

from dataLoad.preprocess import get_data



### 模型可解释性——可视化处理——将特征以及特征融合对模型的分类效果通过散点图的形式展现
def plot_tsne_feature(data_path, feature_path, subject, kfold1, kfold2, kfold3, output_dir, dpi=300, constitute=True):
    '''
    Model Interpretability—Visualization—The feature and classification effect of the feature fusion on the model is displayed in the form of a scatter diagram

    Args:
        data_path: the path of the data
        feature_path: the path of the feature
        constitute: Whether to draw a combination chart (default=True)
    '''
    # 加载数据，获取训练集的标签
    print('Loading data……')
    _, _, _, true_labels, _, _ = get_data(data_path, subject, LOSO=False, LOSO_model='two_session', data_type='2a')
    # 获取所有与特征相关的数据的地址，并将其保存在一个列表
    features_path = []
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'coarseFatures/'))
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'fineFatures/'))
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'fusionFeature/'))
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'EI+MSA+TCN/'))
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'EI+MSA+TCN+FF/'))
    # 读取特征数据和模型分类结果数据
    data = {}
    feature_files = os.listdir(features_path[0])
    for filename  in feature_files:
        if 'fold{}_'.format(kfold1) in filename:
            with open(features_path[0] + filename, "rb") as coarseData:
                coarseFatures = pickle.load(coarseData)
                data['coarseFatures'] = np.reshape(coarseFatures, (coarseFatures.shape[0], -1))
            with open(features_path[1] + filename, "rb") as fineData:
                fineFatures = pickle.load(fineData)
                data['fineFatures'] = np.reshape(fineFatures, (fineFatures.shape[0], -1))
            with open(features_path[2] + filename, "rb") as fusionData:
                fusionFatures = pickle.load(fusionData)
                data['fusionFatures'] = np.reshape(fusionFatures, (fusionFatures.shape[0], -1))
    feature_files = os.listdir(features_path[3])
    for filename  in feature_files:
        if 'fold{}_'.format(kfold2) in filename:
            with open(features_path[3] + filename, "rb") as EI_MSA_TCN_Data:
                EI_MSA_TCN_OUT = pickle.load(EI_MSA_TCN_Data)
                data['EI_MSA_TCN_OUT'] = np.reshape(EI_MSA_TCN_OUT, (EI_MSA_TCN_OUT.shape[0], -1))
    feature_files = os.listdir(features_path[4])
    for filename  in feature_files:
        if 'fold{}_acc'.format(kfold3) in filename:
            with open(features_path[4] + filename, "rb") as EI_MSA_TCN_FF_Data:
                EI_MSA_TCN_FF_OUT = pickle.load(EI_MSA_TCN_FF_Data)
                data['EI_MSA_TCN_FF_OUT'] = np.reshape(EI_MSA_TCN_FF_OUT, (EI_MSA_TCN_FF_OUT.shape[0], -1))
    print('Data loading complete!')

    # 绘图
    print('Image is being generated……')
    data_type = ['coarseFatures', 'fineFatures', 'fusionFatures', 'EI_MSA_TCN_OUT', 'EI_MSA_TCN_FF_OUT']
    labels = ['left hand', 'right hand', 'feet', 'tongue']
    colors = [5, 3, 1, 7]

    if constitute:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        for i in range(5):
            tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=20230520)
            X_tsne = tsne.fit_transform(data[data_type[i]])
            X_tsne = MinMaxScaler().fit_transform(X_tsne)

            for category in np.unique(true_labels):
                axs[i//3, i%3].scatter(
                    *X_tsne[true_labels == category].T, 
                    marker=".", # f"${digit}$",
                    color=plt.cm.Paired(colors[int(category)]),
                    label=labels[int(category)],
                    alpha=0.8,
                    s=100)
            axs[i//3, i%3].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            # axs[i//3, i%3].legend()
        plt.show()
    else:
        output_dir = output_dir + 's{:}/'.format(subject)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=20230520)
        for i in range(1):
            i = 4
            plt.figure(figsize=(5, 5))
            X_tsne = tsne.fit_transform(data[data_type[i]])
            X_tsne = MinMaxScaler().fit_transform(X_tsne)

            for category in np.unique(true_labels):
                plt.scatter(
                    *X_tsne[true_labels == category].T, 
                    marker=".", # f"${digit}$",
                    color=plt.cm.Paired(colors[int(category)]),
                    label=labels[int(category)],
                    alpha=0.8,
                    s=100)
            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.legend()
            output_dir_ = output_dir + "featre_tSNE_11111{:}.png".format(i)
            plt.savefig(output_dir_, dpi=dpi)
            print(f'The picture is saved successfully!\nSave address: '+output_dir_)



### 模型可解释性——可视化处理——将决策以及决策融合对模型的分类效果通过散点图的形式展现
def plot_tsne_decision(data_path, decision_path, subject, kfold1, kfold2, kfold3, output_dir, dpi=300, constitute=True):
    '''
    Model Interpretability—Visualization—The decision and classification effect of the decision fusion on the model is displayed in the form of a scatter diagram

    Args:
        data_path: the path of the data
        decision_path: the path of the decision
        constitute: Whether to draw a combination chart (default=True)
    '''
    # 加载数据，获取训练集的标签
    print('Loading data……')
    _, _, _, true_labels, _, _ = get_data(data_path, subject, LOSO=False, LOSO_model='two_session', data_type='2a')
    # 获取所有与特征相关的数据的地址，并将其保存在一个列表
    decisions_path = []
    decisions_path.append(os.path.join(decision_path, 's{:}'.format(subject), 'eeg_out/'))
    decisions_path.append(os.path.join(decision_path, 's{:}'.format(subject), 'tcn_out/'))
    decisions_path.append(os.path.join(decision_path, 's{:}'.format(subject), 'EI+MSA+TCN/'))
    decisions_path.append(os.path.join(decision_path, 's{:}'.format(subject), 'EI+MSA+TCN+DF/'))
    # 读取特征数据和模型分类结果数据
    data = {}
    decision_files = os.listdir(decisions_path[0])
    for filename  in decision_files:
        if 'fold{}_'.format(kfold1) in filename:
            with open(decisions_path[0] + filename, "rb") as coarseData:
                eeg_out = pickle.load(coarseData)
                data['eeg_out'] = np.reshape(eeg_out, (eeg_out.shape[0], -1))
            with open(decisions_path[1] + filename, "rb") as fineData:
                tcn_out = pickle.load(fineData)
                data['tcn_out'] = np.reshape(tcn_out, (tcn_out.shape[0], -1))
    decision_files = os.listdir(decisions_path[2])
    for filename  in decision_files:
        if 'fold{}_'.format(kfold2) in filename:
            with open(decisions_path[2] + filename, "rb") as EI_MSA_TCN_Data:
                EI_MSA_TCN_OUT = pickle.load(EI_MSA_TCN_Data)
                data['EI_MSA_TCN_OUT'] = np.reshape(EI_MSA_TCN_OUT, (EI_MSA_TCN_OUT.shape[0], -1))
    decision_files = os.listdir(decisions_path[3])
    for filename  in decision_files:
        if 'fold{}_acc'.format(kfold3) in filename:
            with open(decisions_path[3] + filename, "rb") as EI_MSA_TCN_DF_Data:
                EI_MSA_TCN_DF_OUT = pickle.load(EI_MSA_TCN_DF_Data)
                data['EI_MSA_TCN_DF_OUT'] = np.reshape(EI_MSA_TCN_DF_OUT, (EI_MSA_TCN_DF_OUT.shape[0], -1))
    print('Data loading complete!')

    # 绘图
    print('Image is being generated……')
    data_type = ['eeg_out', 'tcn_out', 'EI_MSA_TCN_OUT', 'EI_MSA_TCN_DF_OUT']
    labels = ['left hand', 'right hand', 'feet', 'tongue']
    colors = [5, 3, 1, 7]

    if constitute:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        for i in range(4):
            tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=20230520)
            X_tsne = tsne.fit_transform(data[data_type[i]])
            X_tsne = MinMaxScaler().fit_transform(X_tsne)

            for category in np.unique(true_labels):
                axs[i//2, i%2].scatter(
                    *X_tsne[true_labels == category].T, 
                    marker=".", # f"${digit}$",
                    color=plt.cm.Dark2(int(category)),
                    label=labels[int(category)],
                    alpha=0.8,
                    s=100)
            # ax.set_title(title)
            # axs[i//3, i%3].axis("off")
            axs[i//2, i%2].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            # axs[i//3, i%3].legend()
        plt.show()
    else:
        output_dir = output_dir + 's{:}/'.format(subject)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=20230520)
        for i in range(4):
            plt.figure(figsize=(5, 5))
            X_tsne = tsne.fit_transform(data[data_type[i]])
            X_tsne = MinMaxScaler().fit_transform(X_tsne)

            for category in np.unique(true_labels):
                plt.scatter(
                    *X_tsne[true_labels == category].T, 
                    marker=".", # f"${digit}$",
                    color=plt.cm.Paired(colors[int(category)]),
                    label=labels[int(category)],
                    alpha=0.8,
                    s=100)
            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            # plt.legend()
            output_dir_ = output_dir + "decision_tSNE_{:}.png".format(i)
            plt.savefig(output_dir_, dpi=dpi)
            print(f'The picture is saved successfully!\nSave address: '+output_dir_)



### Class providing methods to carry out filtering of EEG data to be used for FBCSP.
class FilterBank:
    '''
    Class providing methods to carry out filtering of EEG data to be used for FBCSP.

    Args:
        fs: Sampling frequency (int or float)
        f_trans: Transition bandwidth (int or float) (Default=2)
        f_pass: The pass bands of the frequency bands used in FBCSP (numpy.ndarray) (Default=arange(0,40,4))
        f_width: The width of each frequency band (int) (Default=4)
        g_pass: The maximum loss in the passband (dB) (int or float) (Default=3)
        g_stop: The minimum attenuation in the stopband (dB) (int or float) (Default=30)
        filter_coeff: Contains the filter coefficients 'b' and 'a' (dict) (Default={})
    '''
    def __init__(self, fs):
        self.fs = fs
        self.f_trans = 1
        self.f_pass = np.arange(0,120,8)
        self.f_width = 8
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        '''
        Returns the filter coefficients based on filter design parameters.
        '''
        Nyquist_freq = self.fs/2
        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            if (f_pass[0]-self.f_trans) < 0: # 如果是0-4Hz，则使用低通滤波器
                f_pass = f_low_pass+self.f_width
                f_stop = f_pass+self.f_trans
                btype = 'lowpass'
            else: # 否则使用带通滤波器
                f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
                btype = 'bandpass'
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop) # 切比雪夫2型滤波器，返回滤波器阶数
            b, a = signal.cheby2(order, self.gstop, ws, btype=btype) # 切比雪夫2型带通滤波器
            self.filter_coeff.update({i:{'b':b,'a':a}})
        return self.filter_coeff

    def filter_data(self, eeg_data, window_details={}):
        '''
        Returns the filtered data into the various frequency bands for FBCSP.

        Args:
            eeg_data: 3D array (trials, channels, time) containing epoched EEG data.
            window_details: To be used when a smaller window is to be extracted from epoched data after filtering.
            
        Output:
            filtered_data: 4D array (frequency band, trials, channels, time) containing filtered epoched EEG data.
        '''
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))
        filtered_data=np.zeros((len(self.filter_coeff), n_trials, n_channels, n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:,:,int((window_details.get('tmin'))*self.fs):int((window_details.get('tmax'))*self.fs)]
            filtered_data[i,:,:,:]=eeg_data_filtered
        return filtered_data



### 模型可解释性——可视化处理——一键生成时间卷积和空间卷积权重图
def tempDepth_Weight_Visualization(model_path, montage_path, subject, kfold, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
        montage_path: The address of EEG topography
        output_dir: The address where the picture is saved
    '''
    plt.rc('font',family='Times New Roman') # 设置画布的全局字体
    print('Loading model parameters……')
    # 加载模型参数
    model_path = model_path+'s{:}/'.format(subject)
    model_files = os.listdir(model_path)
    for filename  in model_files:
        if 'fold{}_'.format(kfold) in filename:
            file_name = filename
            file_path = os.path.join(model_path, file_name)
            state_dict = torch.load(file_path)
    print('The model parameters are loaded!')
    print('Image is being generated……')
    # 读取时间卷积层和深度卷积的卷积核权重
    temporalConv_wieght = state_dict['sincConv.weight'].cpu().numpy()
    depthConv_wieght = state_dict['conv_depth.weight'].cpu().numpy()
    # 选取有代表性的时间滤波器
    filters = [17, 18, 5, 13, 19, 10, 32, 25]
    # 加载自己修改的（好看的）脑电通道地形图文件
    data1020 = pd.read_excel(montage_path, index_col=0)
    channels1020 = np.array(data1020.index)
    value1020 = np.array(data1020)
    list_dic = dict(zip(channels1020, value1020))
    montage_1020 = mne.channels.make_dig_montage(ch_pos=list_dic,
                                                    nasion=[5.27205792e-18,  8.60992398e-02, -4.01487349e-02],
                                                    lpa=[-0.08609924, -0., -0.04014873],
                                                    rpa=[0.08609924,  0., -0.04014873])
    # 创建info对象
    info = mne.create_info(ch_names=montage_1020.ch_names, sfreq=1., ch_types='eeg')
    # 设置 EEG/sEEG/ECoG/DBS/fNIRS 通道位置和数字化点
    info.set_montage(montage_1020)
    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(nrows=3, ncols=8, figsize=(24, 8))

    # 在画布上绘制8个曲线图
    for i in range(8):
        axs[0, i].plot(np.linspace(0,0.125,32,endpoint=True), temporalConv_wieght[filters[i]-1,0,0,:])
        axs[0, i].set_title("Temp. Filter {}".format(i+1), fontdict={'size':16, 'weight':'bold'}) # 
        if i==0:
            axs[0, i].set_ylim(-0.2,0.2) # 设置y轴的范围
            axs[0, i].set_xticks([0, 0.125]) # 设置在哪个地方显示刻度
            axs[0, i].set_xticklabels(['0','0.125']) # 设置显示刻度的内容
            axs[0, i].tick_params(labelsize=14) # 设置刻度字体大小
        else:
            axs[0, i].set_ylim(-0.2,0.2) # 设置y轴的范围
            axs[0, i].set_xticks([0, 0.125]) # 设置在哪个地方显示刻度
            axs[0, i].set_xticklabels(['0','0.125'], fontdict={'size': 14}) # 设置显示刻度的内容
            axs[0, i].tick_params(axis='both', which='both', left=False, labelleft=False) # 不显示y轴的内容

    # 在画布上绘制8个曲线图对应的地形图
    for i in range(8):
        mne.viz.plot_topomap(depthConv_wieght[(filters[i]-1)*2,0,:,0], info, show=False, axes=axs[1,i], extrapolate='head')
        mne.viz.plot_topomap(depthConv_wieght[(filters[i]-1)*2+1,0,:,0], info, show=False, axes=axs[2,i], extrapolate='head')
        if i==0:
            axs[1, i].set_ylabel('Spat. Filter 1', fontdict={'size': 16, 'weight':'bold'}) # 设置y轴的标签
            axs[2, i].set_ylabel('Spat. Filter 2', fontdict={'size': 16, 'weight':'bold'}) # 设置y轴的标签
    # 创建图片保存的地址
    output_dir = output_dir + 's{:}/'.format(subject)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = output_dir + 'TempDepth Weight Visualization.png'
    plt.savefig(output_dir, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: '+output_dir)
    plt.show()



### 模型可解释性——可视化处理——一键生成Inception块的时间卷积权重图
def Inception_Weight_Visualization(model_path, subject, kfold, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
        output_dir: The address where the picture is saved
    '''
    plt.rc('font',family='Times New Roman') # 设置画布的全局字体
    print('Loading model parameters……')
    # 加载模型参数
    model_path = model_path+'s{:}/'.format(subject)
    model_files = os.listdir(model_path)
    for filename  in model_files:
        if 'fold{}_'.format(kfold) in filename:
            file_name = filename
            file_path = os.path.join(model_path, file_name)
            state_dict = torch.load(file_path)
    print('The model parameters are loaded!')
    print('Image is being generated……')
    # 读取时间卷积层的卷积核权重
    temporalConv_wieght = []
    temporalConv_wieght.append(state_dict['incept_temp.conv3.weight'].cpu().numpy()) # 0.125
    temporalConv_wieght.append(state_dict['incept_temp.conv2.weight'].cpu().numpy()) # 0.25
    temporalConv_wieght.append(state_dict['incept_temp.conv1.weight'].cpu().numpy()) # 0.5
    # 选取要绘制的滤波器
    filters = [[1, 4, 10, 18, 42, 54, 57, 62],
               [33, 36, 39, 43, 51, 54, 58, 63], 
               [1, 6, 7, 11, 21, 33, 41, 44]]
    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(nrows=3, ncols=8, figsize=(24, 8))
    # 对三个卷积的权重进行循环绘图
    for conv in range(3):
        time_end = 0.125*(2**conv)
        num_point = 4*(2**conv)
        # 在画布上绘制8个地形图
        for i in range(8):
            axs[conv, i].plot(np.linspace(0,time_end,num_point,endpoint=True), temporalConv_wieght[conv][(filters[conv][i]-1)//4,(filters[conv][i]-1)%4,0,:])
            if conv == 0:
                axs[conv, i].set_title("Temp. Filter {}".format(i+1), fontdict={'size':16, 'weight':'bold'}) # 
            if i == 0:
                axs[conv, i].set_ylim(-0.2,0.2) # 设置y轴的范围
                axs[conv, i].set_ylabel('DW Conv {}'.format(conv+1), fontdict={'size': 16, 'weight':'bold'}) # 设置y轴的标签
                axs[conv, i].set_xticks([0, time_end]) # 设置在哪个地方显示刻度
                axs[conv, i].set_xticklabels(['0',str(time_end)]) # 设置显示刻度的内容
                axs[conv, i].tick_params(labelsize=12) # 设置刻度字体大小
            else:
                axs[conv, i].set_ylim(-0.2,0.2) # 设置y轴的范围
                axs[conv, i].set_xticks([0, time_end]) # 设置在哪个地方显示刻度
                axs[conv, i].set_xticklabels(['0',str(time_end)], fontdict={'size': 14}) # 设置显示刻度的内容
                axs[conv, i].tick_params(axis='both', which='both', left=False, labelleft=False) # 不显示y轴的内容
    # 创建图片保存的地址
    output_dir = output_dir + 's{:}/'.format(subject)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = output_dir + 'Inception Weight Visualization.png'
    plt.savefig(output_dir, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: '+output_dir)
    plt.show()



### 模型可解释性——可视化处理——一键生成FBCSP和深度卷积权重对比图
def FBCSP_Weight_Visualization(data_path, model_path, subject, kfold, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
        output_dir: The address where the picture is saved
    '''
    plt.rc('font',family='Times New Roman') # 设置画布的全局字体
    # 禁用 MNE 输出信息到终端
    mne.set_log_level('WARNING')
    print('Loading data……')
    # 加载数据
    data, labels, _, _, _, _ = get_data(data_path, subject, LOSO=False, LOSO_model='two_session', data_type='2a')
    print('Data loading complete!')
    print('Band processing in progress……')
    # 对数据进行分频带处理
    fbank = FilterBank(fs)
    fbank_coeff = fbank.get_filter_coeff()
    filtered_data = fbank.filter_data(data, window_details)
    print('Band processing complete!')
    # 获取分频带后的数据的尺寸
    shape = filtered_data.shape #  (frequency_bands, n_trials, n_channels, n_samples)
    # 加载自己修改的（好看的）脑电通道地形图文件
    data1020 = pd.read_excel(montage_path, index_col=0)
    channels1020 = np.array(data1020.index)
    value1020 = np.array(data1020)
    list_dic = dict(zip(channels1020, value1020))
    montage_1020 = mne.channels.make_dig_montage(ch_pos=list_dic,
                                                nasion=[5.27205792e-18,  8.60992398e-02, -4.01487349e-02],
                                                lpa=[-0.08609924, -0., -0.04014873],
                                                rpa=[0.08609924,  0., -0.04014873])
    # 创建info实例
    info = mne.create_info(ch_names=montage_1020.ch_names, sfreq=1., ch_types='eeg')
    info.set_montage(montage_1020)
    # 创建图片保存的地址
    output_dir = output_dir + 's{:}/'.format(subject)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ## 绘制FBCSP的滤波器图
    # 创建CSP对象
    csp = CSP(n_components, reg=None, log=True, norm_trace=False)
    # 选取要计算的频带
    fbanks = [2, 3, 7, 9]
    filters = [[1,2], [2,3], [1,2], [1,2]]
    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(4, 2, figsize=(4, 8))
    print('Image is being generated……')
    for fb in range(4):
        data = filtered_data[fbanks[fb]-1]
        csp.fit(data, labels)
        filter = csp.filters_.T
        for n_filter in range(0, 2):
            mne.viz.plot_topomap(filter[:, filters[fb][n_filter]-1], info, show=False, axes=axs[fb, n_filter], extrapolate='head')
            if fb == 0:
                axs[fb, n_filter].set_title('Spatial Filter {}'.format(n_filter+1), fontdict={'size':16})
            if n_filter == 0:
                axs[fb, n_filter].set_ylabel('{}-{}Hz'.format((fbanks[fb]-1)*8, (fbanks[fb])*8), fontdict={'size': 16}) # 设置y轴的标签
    # 保存图片
    output_dir_ = output_dir + 'FBCSP_Weight_1.png'
    plt.savefig(output_dir_, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: '+output_dir_)
    ## 绘制深度卷积权重图
    print('Loading model parameters……')
    # 加载模型参数
    model_path = model_path+'s{:}/'.format(subject)
    model_files = os.listdir(model_path)
    for filename  in model_files:
        if 'fold{}_'.format(kfold) in filename:
            file_name = filename
            file_path = os.path.join(model_path, file_name)
            state_dict = torch.load(file_path)
    print('The model parameters are loaded!')
    print('Image is being generated……')
    # 读取深度卷积的卷积核权重
    depthConv_wieght = state_dict['conv_depth.weight'].cpu().numpy()
    # 选取有代表性的时间滤波器
    filters = [17, 18, 32, 25]
    frequency = [2*8, 3*8, 7*8, 9*8]
    temp_filter = [1, 2, 7, 8]
    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(4, 8))
    # 在画布上绘制8个曲线图对应的地形图
    for i in range(4):
        mne.viz.plot_topomap(depthConv_wieght[(filters[i]-1)*2,0,:,0], info, show=False, axes=axs[i,0], extrapolate='head')
        mne.viz.plot_topomap(depthConv_wieght[(filters[i]-1)*2+1,0,:,0], info, show=False, axes=axs[i,1], extrapolate='head')
        if i==0:
            axs[i, 0].set_title('Spatial Filter 1', fontdict={'size': 16}) # 设置第一行的标题
            axs[i, 1].set_title('Spatial Filter 2', fontdict={'size': 16}) # 设置第一行的标题
        axs[i, 0].set_ylabel('{}Hz(Temp. Filter {})'.format(frequency[i], temp_filter[i]), fontdict={'size': 12}) # 设置y轴的标签
    # 保存图片
    output_dir_ = output_dir + 'FBCSP_Weight_2.png'
    plt.savefig(output_dir_, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: '+output_dir_)



### 模型可解释性——可视化处理——所有被试的混淆矩阵
def  confusion_matrix_disply(confusion, labels, output_dir, dataset='2a', dpi=300):
    '''
    Model Interpretability—Visualization—Confusion matrix of all subjects.

    Args:
        confusion: Confusion matrix
        labels: The  labels of the classes
        output_dir: The address where the image is saved
        dataset: Data set to plot Confusion matrix
    '''
     
    plt.rc('font',family='Times New Roman') # 设置画布的全局字体

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion)

    # Plot the confusion matrix and get the Axes object
    ax = disp.plot(include_values=True, cmap=plt.cm.Reds, values_format='.2%', colorbar=True, ax=None)

    # Add minor ticks at each cell boundary
    x_ticks = np.arange(-.5, len(labels), 1)
    y_ticks = np.arange(-.5, len(labels), 1)
    ax.ax_.set_xticks(x_ticks, minor=True)
    ax.ax_.set_yticks(y_ticks, minor=True)

    ax.ax_.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False,
                                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # Add grid for minor ticks
    ax.ax_.grid(True, which='minor', color='white', linestyle='-', linewidth=1.5)

    ax.ax_.set_yticklabels(labels, va='center')
    ax.ax_.set_xticklabels(labels, va='center')
    ax.ax_.xaxis.set_tick_params(pad=10)

    plt.yticks(rotation=90)

    output_dir = output_dir + 'Confusion Matrix '+dataset+'.png'
    plt.savefig(output_dir, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: '+output_dir)

    plt.show()



###============================ Initialization parameters ============================###
subject = 1
kfold = 4

data_path = "/home/pytorch/LiangXiaohan/MI_Dataverse/BCICIV_2a/mat/"
model_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Ablation study/7 EI+MSA+TCN+FM/"
montage_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/1020-22.xlsx"
output_dir = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/weight visualization/"
feature_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/t-sne/feature/"
decision_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/t-sne/decision/"

fs = 250
window_details = {'tmin':0,'tmax':4}
n_components = 4

output_dir_cm = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/"

confusion_2a = np.array([[0.8796, 0.0540, 0.0417, 0.0247],
                         [0.0633 , 0.8534, 0.0509, 0.0324],
                         [0.0540, 0.0679, 0.7978, 0.0802],
                         [0.0864, 0.0725, 0.0448, 0.7963 ]])
labels_2a = ['Left', 'Right', 'Foot', 'Tongue']

confusion_2b = np.array([[0.8785, 0.1076],
                         [0.1354, 0.8507]])
labels_2b = ['Left Hand', 'Right Hand']

###============================ Main function ============================###
def main():

    plot_tsne_feature(data_path, feature_path, subject, kfold1=2, kfold2=1, kfold3=3, output_dir=feature_path, dpi=600, constitute=False)

    plot_tsne_decision(data_path, decision_path, subject, kfold1=1, kfold2=1, kfold3=1, output_dir=decision_path, dpi=600, constitute=False)

    tempDepth_Weight_Visualization(model_path, montage_path, subject, kfold, output_dir, dpi=600)

    Inception_Weight_Visualization(model_path, subject, kfold, output_dir, dpi=600)

    FBCSP_Weight_Visualization(data_path, model_path, subject, kfold, output_dir, dpi=600)

    confusion_matrix_disply(confusion_2a, labels_2a, output_dir_cm, dataset='2a', dpi=600)

    confusion_matrix_disply(confusion_2b, labels_2b, output_dir_cm, dataset='2b', dpi=600)

if __name__ == "__main__":
    main()
