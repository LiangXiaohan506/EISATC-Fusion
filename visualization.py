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
from sklearn.metrics import accuracy_score

from dataLoad.preprocess import get_data
from functions import getModel



### 模型可解释性——可视化处理——将特征以及特征融合对模型的分类效果通过散点图的形式展现
def plot_tsne_feature(data_path, feature_path, subject, kfoldNum, output_dir, dpi=300, constitute=True):
    '''
    Model Interpretability—Visualization—The feature and classification effect of the feature fusion on the model is displayed in the form of a scatter diagram

    Args:
        data_path: the path of the data
        feature_path: the path of the feature
        constitute: Whether to draw a combination chart (default=True)
    '''
    # 图片输出路径
    output_dir = output_dir + 's{:}/'.format(subject) + 't-sne figure/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 加载数据，获取训练集的标签
    print('Loading data……')
    _, _, _, true_labels, _, _ = get_data(data_path, subject, LOSO=False, data_model='one_session', data_type='2a_mat', start_time=2, end_time=6)
    # 获取所有与特征相关的数据的地址，并将其保存在一个列表
    features_path = []
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'coarseFatures/'))
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'fineFatures/'))
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'fusionFeature/'))
    # 读取特征数据和模型分类结果数据
    feature_files = os.listdir(features_path[0])
    for kfold in range(kfoldNum):
        data = {}
        for filename  in feature_files:
            if 'fold{}_'.format(kfold) in filename:
                with open(features_path[0] + filename, "rb") as coarseData:
                    coarseFatures = pickle.load(coarseData)
                    data['coarseFatures'] = np.reshape(coarseFatures, (coarseFatures.shape[0], -1))
                with open(features_path[1] + filename, "rb") as fineData:
                    fineFatures = pickle.load(fineData)
                    data['fineFatures'] = np.reshape(fineFatures, (fineFatures.shape[0], -1))
                with open(features_path[2] + filename, "rb") as fusionData:
                    fusionFatures = pickle.load(fusionData)
                    data['fusionFatures'] = np.reshape(fusionFatures, (fusionFatures.shape[0], -1))
        print('Data loading complete!')

        # 绘图
        print('Image is being generated……')
        data_type = ['coarseFatures', 'fineFatures', 'fusionFatures']
        labels = ['left hand', 'right hand', 'feet', 'tongue']
        colors = [5, 3, 1, 7]

        if constitute:
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
            for i in range(3):
                tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=202411103)
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
            tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=202411103)
            for i in range(3):
                plt.figure(figsize=(5, 5))
                X_tsne = tsne.fit_transform(data[data_type[i]])
                X_tsne = MinMaxScaler().fit_transform(X_tsne)

                for category in np.unique(true_labels):
                    plt.scatter(
                        *X_tsne[true_labels == category].T, 
                        marker=".", # f"${digit}$",
                        color=plt.cm.Paired(colors[int(category)]),
                        # label=labels[int(category)],
                        alpha=0.8,
                        s=100)
                plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                plt.legend()
                output_dir_ = output_dir + "featre_tSNE_kfold{:}_{:}.png".format(kfold, i)
                plt.savefig(output_dir_, dpi=dpi)
                print(f'The picture is saved successfully!\nSave address: '+output_dir_)


### 模型可解释性——可视化处理——将决策以及决策融合对模型的分类效果通过散点图的形式展现
def plot_tsne_decision(data_path, decision_path, subject, kfoldNum, output_dir, dpi=300, constitute=True):
    '''
    Model Interpretability—Visualization—The decision and classification effect of the decision fusion on the model is displayed in the form of a scatter diagram

    Args:
        data_path: the path of the data
        decision_path: the path of the decision
        constitute: Whether to draw a combination chart (default=True)
    '''
    # 图片输出路径
    output_dir = output_dir + 's{:}/'.format(subject) + 't-sne figure/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 加载数据，获取训练集的标签
    print('Loading data……')
    _, _, _, true_labels, _, _ = get_data(data_path, subject, LOSO=False, data_model='one_session', data_type='2a_mat', start_time=2, end_time=6)
    # 获取所有与特征相关的数据的地址，并将其保存在一个列表
    decisions_path = []
    decisions_path.append(os.path.join(decision_path, 's{:}'.format(subject), 'eeg_out/'))
    decisions_path.append(os.path.join(decision_path, 's{:}'.format(subject), 'tcn_out/'))
    decisions_path.append(os.path.join(decision_path, 's{:}'.format(subject), 'fusionOutput/'))
    # 读取特征数据和模型分类结果数据
    decision_files = os.listdir(decisions_path[0])
    for kfold in range(kfoldNum):
        data = {}
        for filename  in decision_files:
            if 'fold{}_'.format(kfold) in filename:
                with open(decisions_path[0] + filename, "rb") as coarseData:
                    eeg_out = pickle.load(coarseData)
                    data['eeg_out'] = np.reshape(eeg_out, (eeg_out.shape[0], -1))
                with open(decisions_path[1] + filename, "rb") as fineData:
                    tcn_out = pickle.load(fineData)
                    data['tcn_out'] = np.reshape(tcn_out, (tcn_out.shape[0], -1))
                with open(decisions_path[2] + filename, "rb") as fusionData:
                    fusionOutput = pickle.load(fusionData)
                    data['fusionData'] = np.reshape(fusionOutput, (fusionOutput.shape[0], -1))
        print('Data loading complete!')

        # 绘图
        print('Image is being generated……')
        data_type = ['eeg_out', 'tcn_out', 'fusionData']
        labels = ['left hand', 'right hand', 'feet', 'tongue']
        colors = [5, 3, 1, 7]

        if constitute:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
            for i in range(3):
                tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=202411103)
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
            tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=202411103)
            for i in range(3):
                plt.figure(figsize=(5, 5))
                X_tsne = tsne.fit_transform(data[data_type[i]])
                X_tsne = MinMaxScaler().fit_transform(X_tsne)

                for category in np.unique(true_labels):
                    plt.scatter(
                        *X_tsne[true_labels == category].T, 
                        marker=".", # f"${digit}$",
                        color=plt.cm.Paired(colors[int(category)]),
                        # label=labels[int(category)],
                        alpha=0.8,
                        s=100)
                plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                output_dir_ = output_dir + "featre_tSNE_kfold{:}_{:}.png".format(kfold, i)
                plt.savefig(output_dir_, dpi=dpi)
                print(f'The picture is saved successfully!\nSave address: '+output_dir_)


### 模型可解释性——可视化处理——模型输出通过散点图的形式展现，证明特征融合和决策融合
def plot_tsne_output(data_path, feature_path, subject, kfoldNum, output_dir, dpi=300, constitute=True):
    '''
    Model Interpretability—Visualization—The feature and classification effect of the feature fusion on the model is displayed in the form of a scatter diagram

    Args:
        data_path: the path of the data
        feature_path: the path of the feature
        constitute: Whether to draw a combination chart (default=True)
    '''
    # 图片输出路径
    output_dir_list = []
    model_type = ['EI+MSA+TCN', 'EI+MSA+TCN+DF', 'EI+MSA+TCN+FF', 'EI+MSA+TCN+FM']
    for i in range(4):
        output_dir_list.append(output_dir + 's{:}/'.format(subject) + model_type[i] + '/t-sne figure/')
        if not os.path.exists(output_dir_list[i]):
            os.makedirs(output_dir_list[i])
    # 加载数据，获取训练集的标签
    print('Loading data……')
    _, _, _, true_labels, _, _ = get_data(data_path, subject, LOSO=False, data_model='one_session', data_type='2a_mat', start_time=2, end_time=6)
    # 获取所有与特征相关的数据的地址，并将其保存在一个列表
    features_path = []
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'EI+MSA+TCN/'))
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'EI+MSA+TCN+DF/'))
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'EI+MSA+TCN+FF/'))
    features_path.append(os.path.join(feature_path, 's{:}'.format(subject), 'EI+MSA+TCN+FM/'))
    # 读取特征数据和模型分类结果数据
    feature_files_0 = os.listdir(features_path[0])
    feature_files_1 = os.listdir(features_path[1])
    feature_files_2 = os.listdir(features_path[2])
    feature_files_3 = os.listdir(features_path[3])
    for kfold in range(kfoldNum):
        data = {}
        for filename in feature_files_0:
            if 'fold{}_'.format(kfold) in filename:
                with open(features_path[0] + filename, "rb") as EI_MSA_TCN_Data:
                    EI_MSA_TCN_OUT = pickle.load(EI_MSA_TCN_Data)
                    data['EI_MSA_TCN_OUT'] = np.reshape(EI_MSA_TCN_OUT, (EI_MSA_TCN_OUT.shape[0], -1))
        for filename in feature_files_1:
            if 'fold{}_'.format(kfold) in filename:
                with open(features_path[1] + filename, "rb") as EI_MSA_TCN_DF_Data:
                    EI_MSA_TCN_DF_OUT = pickle.load(EI_MSA_TCN_DF_Data)
                    data['EI_MSA_TCN_DF_OUT'] = np.reshape(EI_MSA_TCN_DF_OUT, (EI_MSA_TCN_DF_OUT.shape[0], -1))
        for filename in feature_files_2:
            if 'fold{}_'.format(kfold) in filename:
                with open(features_path[2] + filename, "rb") as EI_MSA_TCN_FF_Data:
                    EI_MSA_TCN_FF_OUT = pickle.load(EI_MSA_TCN_FF_Data)
                    data['EI_MSA_TCN_FF_OUT'] = np.reshape(EI_MSA_TCN_FF_OUT, (EI_MSA_TCN_FF_OUT.shape[0], -1))
        for filename in feature_files_3:
            if 'fold{}_'.format(kfold) in filename:
                with open(features_path[3] + filename, "rb") as EI_MSA_TCN_FM_Data:
                    EI_MSA_TCN_FM_OUT = pickle.load(EI_MSA_TCN_FM_Data)
                    data['EI_MSA_TCN_FM_OUT'] = np.reshape(EI_MSA_TCN_FM_OUT, (EI_MSA_TCN_FM_OUT.shape[0], -1))
        print('Data loading complete!')

        # 绘图
        print('Image is being generated……')
        data_type = ['EI_MSA_TCN_OUT', 'EI_MSA_TCN_DF_OUT', 'EI_MSA_TCN_FF_OUT', 'EI_MSA_TCN_FM_OUT']
        labels = ['left hand', 'right hand', 'feet', 'tongue']
        colors = [5, 3, 1, 7]

        if constitute:
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
            for i in range(4):
                tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=202411103)
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
            tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=202411103)
            for i in range(4):
                plt.figure(figsize=(5, 5))
                X_tsne = tsne.fit_transform(data[data_type[i]])
                X_tsne = MinMaxScaler().fit_transform(X_tsne)

                for category in np.unique(true_labels):
                    plt.scatter(
                        *X_tsne[true_labels == category].T, 
                        marker=".", # f"${digit}$",
                        color=plt.cm.Paired(colors[int(category)]),
                        # label=labels[int(category)],
                        alpha=0.8,
                        s=100)
                plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                plt.legend()
                output_dir_ = output_dir_list[i] + "featre_tSNE_kfold{:}.png".format(kfold)
                plt.savefig(output_dir_, dpi=dpi)
                print(f'The picture is saved successfully!\nSave address: '+output_dir_)


### 模型可解释性——可视化处理——将时间卷积层的卷积核权重系数用脑电地形图的方式展现出来
def inceptionWeight_to_waveform(model_path, subject, kfoldNum, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
    '''
    # 创建图片保存的地址
    output_dir = output_dir + 's{:}/'.format(subject) + 'Inception Weight/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('Loading model parameters……')
    # 加载模型参数
    model_path = model_path + 's{:}/'.format(subject)
    model_files = os.listdir(model_path)
    for kfold in range(kfoldNum):
        for filename  in model_files:
            if 'fold{}_'.format(kfold) in filename:
                file_name = filename
                file_path = os.path.join(model_path, file_name)
                state_dict = torch.load(file_path)
        print('The model parameters are loaded!')
        print('Image is being generated……')
        # 读取时间卷积层的卷积核权重
        temporalConv_wieght = []
        temporalConv_wieght.append(state_dict['incept_temp.conv1.weight'].cpu().numpy())
        temporalConv_wieght.append(state_dict['incept_temp.conv2.weight'].cpu().numpy())
        temporalConv_wieght.append(state_dict['incept_temp.conv3.weight'].cpu().numpy())
        # 对三个卷积的权重进行循环绘图
        for conv in range(3):
            # 设置画布行列数和画布的尺寸
            fig, axs = plt.subplots(nrows=4, ncols=8, figsize=(16, 8))
            # 在画布上绘制32个地形图
            for i in range(32):
                axs[i%4, i//4].plot(temporalConv_wieght[conv][i//4,i%4,0,:])
                # axs[i//8, i%8].set_xlim(0,0.125)
                axs[i%4, i//4].set_ylim(-0.2,0.2)
            output_dir_ = output_dir + "waveform_kfold{:}_conv{:}.png".format(kfold, conv)
            plt.savefig(output_dir_, dpi=dpi)
            print(f'The picture is saved successfully!\nSave address: '+output_dir_)
        plt.show()



### 模型可解释性——可视化处理——将cnnCosMSA的注意力信息显示出来
def attention_weight_visualization(attention_path, subject, kfold, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
        output_dir: The address where the picture is saved
    '''
    # 图片输出路径
    output_dir = output_dir + 's{:}/'.format(subject) + 'attention figure/kFold_{:}/'.format(kfold)
    # 获取所有与特征相关的数据的地址，并将其保存在一个列表
    features_path = []
    # features_path.append(os.path.join(attention_path, 's{:}'.format(subject), 'attention cycle/'))
    features_path.append(os.path.join(attention_path, 's{:}'.format(subject), 'attention value/'))
    # features_path.append(os.path.join(attention_path, 's{:}'.format(subject), 'cycle_attn value/'))
    # features_path.append(os.path.join(attention_path, 's{:}'.format(subject), 'headsita cycle/'))
    # 读取特征数据和模型分类结果数据
    data = {}
    feature_files = os.listdir(features_path[0])
    for filename  in feature_files:
        if 'fold{}_'.format(kfold) in filename:
            with open(features_path[0] + filename, "rb") as attncycleData:
                data['attnCycle'] = pickle.load(attncycleData)

    # feature_files = os.listdir(features_path[1])
    # for filename  in feature_files:
    #     if 'fold{}_'.format(kfold) in filename: 
    #         with open(features_path[1] + filename, "rb") as attnvalueData:
    #             data['attnValue'] = pickle.load(attnvalueData)

    # feature_files = os.listdir(features_path[2])
    # for filename  in feature_files:
    #     if 'fold{}_'.format(kfold) in filename: 
    #         with open(features_path[2] + filename, "rb") as cycleValueData:
    #             data['cycleValue'] = pickle.load(cycleValueData)

    # feature_files = os.listdir(features_path[3])
    # for filename  in feature_files:
    #     if 'fold{}'.format(kfold) in filename:
    #         with open(features_path[3] + filename, "rb") as headcycleData:
    #             headCycle = pickle.load(headcycleData)
    #             headCycle_f = headCycle*(15-1)+1 # 其中的15根据情况更改
    #             data['headCycle'] = headCycle_f

    print('Data loading complete!')

    for head in range(data['attnCycle'].shape[1]):
        for i in range(data['attnCycle'].shape[0]//8): # 绘制288个图像，具体绘制多少自己把握  , (data['attnCycle'].shape[0]//8)*2
            im_Cycle = plt.imshow(data['attnCycle'][i][head], cmap=plt.cm.Reds, vmax=1., vmin=-1.)
            plt.colorbar(im_Cycle)
            plt.tight_layout()
            output_dir_ = output_dir +  "headFreq_{:4f}/attnCycle/".format(head) # data['headCycle'][head])
            if not os.path.exists(output_dir_):
                os.makedirs(output_dir_)
            output_dir_Cycle = output_dir_ + "kfold{:}_{:}.png".format(kfold, i)
            plt.savefig(output_dir_Cycle, dpi=dpi)
            print(f'The picture is saved successfully!\nSave address: ' + output_dir_Cycle)
            plt.clf()

            # im_Value = plt.imshow(data['attnValue'][i][head], cmap=plt.cm.Reds, vmax=1., vmin=-1.)
            # plt.colorbar(im_Value)
            # plt.tight_layout()
            # output_dir_ = output_dir +  "headFreq_{:.5}/attnValue/".format(data['headCycle'][head])
            # if not os.path.exists(output_dir_):
            #     os.makedirs(output_dir_)
            # output_dir_Value = output_dir_ + "kfold{:}_{:}.png".format(kfold, i)
            # plt.savefig(output_dir_Value, dpi=dpi)
            # print(f'The picture is saved successfully!\nSave address: ' + output_dir_Value)
            # plt.clf()

        # im_Value = plt.imshow(data['cycleValue'][head], cmap=plt.cm.Reds, vmax=1., vmin=0.)
        # plt.colorbar(im_Value)
        # plt.tight_layout()
        # output_dir_ = output_dir +  "headFreq_{:4f}/cycleValue/".format(data['headCycle'][head])
        # if not os.path.exists(output_dir_):
        #     os.makedirs(output_dir_)
        # output_dir_cycleValue = output_dir_ + "kfold{:}.png".format(kfold)
        # plt.savefig(output_dir_cycleValue, dpi=dpi)
        # print(f'The picture is saved successfully!\nSave address: ' + output_dir_cycleValue)
        # plt.clf()


def Attention_Weight_Visualization_hybrid(attention_path, subject, kfold, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
        output_dir: The address where the picture is saved
    '''
    plt.rc('font',family='Times New Roman') # 设置画布的全局字体
    # 图片输出路径
    output_dir = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/EEGInc cnnMSA TCN FM/attention/"
    output_dir = output_dir + 's{:}/'.format(subject)
    # 读取特征数据和模型分类结果数据
    data = {}

    features_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/attention/s{:}/".format(subject)
    feature_files = os.listdir(features_path)
    for filename  in feature_files:
        if 'fold{}_'.format(kfold) in filename:
            with open(features_path + filename, "rb") as attnData:
                data_1 = pickle.load(attnData)
                data['MSA'] = np.reshape(data_1, (-1, 8, data_1.shape[1], data_1.shape[2]))

    features_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/ATCNet/Visualization/attention/s{:}/attention value/".format(subject)
    feature_files = os.listdir(features_path)
    for filename  in feature_files:
        if 'fold{}_'.format(kfold) in filename:
            with open(features_path + filename, "rb") as attncycleData:
                data['ATCNet'] = pickle.load(attncycleData)

    features_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/EEG-Conformer-main/Attention/s{:}/".format(subject)
    feature_files = os.listdir(features_path)
    data_eeg = []
    for layer in range(6):
        for filename in feature_files:
            if 'layer_{}'.format(layer+1) in filename:
                with open(features_path + filename, "rb") as attncycleData:
                    data_eeg.append(pickle.load(attncycleData))
    data['EEGConformer'] = data_eeg

    print('Data loading complete!')

    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(nrows=2, ncols=8, figsize=(18, 6))
    samples_idx = np.random.randint(data['ATCNet'].shape[0]//8, size=8)
    head_idx = np.random.randint(data['EEGConformer'][0].shape[1], size=6)
    ATCNet_idx = [0, 10]

    for head in range(8): # 列
        im = axs[0][head].imshow(data['MSA'][samples_idx[head]][head], cmap=plt.cm.Reds, vmax=1., vmin=-1.)
        axs[0][head].set_title("head {:}".format(head+1), fontdict={'size':16, 'weight':'bold'})

    for head in range(2): # 列
        im = axs[1][head].imshow(data['ATCNet'][ATCNet_idx[head]][head], cmap=plt.cm.Reds, vmax=1., vmin=-1.)
        axs[1][head].set_title("head {:}".format(head+1), fontdict={'size':16, 'weight':'bold'})

    for layer in range(6): # 列
        im = axs[1][layer+2].imshow(data['EEGConformer'][layer][samples_idx[head]][head_idx[layer]], cmap=plt.cm.Reds, vmax=1., vmin=-1.)
        axs[1][layer+2].set_title("layer {:}".format(layer+1), fontdict={'size':16, 'weight':'bold'})

        # if head == 7:
        #     fig.colorbar(im, ax=axs[head])

    # plt.show()
    output_dir_cycleValue = output_dir + "kfold{:}.png".format(kfold)
    plt.savefig(output_dir_cycleValue, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: ' + output_dir_cycleValue)


### 模型可解释性——可视化处理——一键生成cnnCosMSA的注意力信息效果图
def Attention_Weight_Visualization(attention_path, subject, kfold, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
        output_dir: The address where the picture is saved
    '''
    plt.rc('font',family='Times New Roman') # 设置画布的全局字体
    # 图片输出路径
    output_dir = output_dir + 's{:}/'.format(subject) + 'attention figure/kFold_{:}/'.format(kfold)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 获取所有与特征相关的数据的地址，并将其保存在一个列表
    features_path = []
    features_path.append(os.path.join(attention_path, 's{:}'.format(subject), 'attention cycle/'))
    features_path.append(os.path.join(attention_path, 's{:}'.format(subject), 'attention value/'))
    features_path.append(os.path.join(attention_path, 's{:}'.format(subject), 'cycle_attn value/'))
    features_path.append(os.path.join(attention_path, 's{:}'.format(subject), 'headsita cycle/'))
    # 读取特征数据和模型分类结果数据
    data = {}
    feature_files = os.listdir(features_path[0]) #
    for filename  in feature_files:
        if 'fold{}_'.format(kfold) in filename:
            with open(features_path[0] + filename, "rb") as attncycleData:
                data['CosATscores'] = pickle.load(attncycleData)
    feature_files = os.listdir(features_path[1]) #
    for filename  in feature_files:
        if 'fold{}_'.format(kfold) in filename: 
            with open(features_path[1] + filename, "rb") as attnvalueData:
                data['ATscores'] = pickle.load(attnvalueData)
    feature_files = os.listdir(features_path[2]) #
    for filename  in feature_files:
        if 'fold{}_'.format(kfold) in filename: 
            with open(features_path[2] + filename, "rb") as cycleValueData:
                data['CosAT'] = pickle.load(cycleValueData)
    feature_files = os.listdir(features_path[3]) #
    for filename  in feature_files:
        if 'fold{}'.format(kfold) in filename:
            with open(features_path[3] + filename, "rb") as headcycleData:
                headCycle = pickle.load(headcycleData)
                headCycle_f = headCycle*(15-1)+1 # 其中的15根据情况更改
                data['headCycle'] = headCycle_f
    print('Data loading complete!')

    # 选取要绘制的滤波器
    samples = [1, 0, 68, 42, 27, 70, 26, 24]
    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(nrows=3, ncols=8, figsize=(20, 7))

    heads = data['headCycle'].shape[0]
    attnClasses = ['ATscores', 'CosAT', 'CosATscores']

    for attnCla in range(len(attnClasses)): # 行
        for head in range(heads): # 列

            if attnCla != 1:
                axs[attnCla, head].imshow(data[attnClasses[attnCla]][samples[head]][head], cmap=plt.cm.Reds, vmax=1., vmin=-1.)
            else:
                axs[attnCla, head].imshow(data[attnClasses[attnCla]][head], cmap=plt.cm.Reds, vmax=1., vmin=0.)

            if attnCla == 0:
                axs[attnCla, head].set_title("head {:}\n$\omega\'$ = {:.5}".format(head+1, data['headCycle'][head]), fontdict={'size':16, 'weight':'bold'}) # 
            if head == 0:
                axs[attnCla, head].set_ylabel(attnClasses[attnCla], fontdict={'size': 16, 'weight':'bold'}) # 设置y轴的标签

            # if head == heads-1:
            #     fig.colorbar(im, ax=axs[attnCla, head])

    # plt.show()
    output_dir_cycleValue = output_dir + "kfold{:}_title.png".format(kfold)
    plt.savefig(output_dir_cycleValue, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: ' + output_dir_cycleValue)
            

### 模型可解释性——可视化处理——将MSA的注意力信息显示出来
def attention_weight_visualization_bef(attention_path, subject, kfold, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
        output_dir: The address where the picture is saved
    '''
    # 图片输出路径
    output_dir = output_dir + 's{:}/'.format(subject) + 'attention figure/kFold_{:}/'.format(kfold)
    # 获取所有与特征相关的数据的地址，并将其保存在一个列表
    features_path = attention_path + 's{:}/'.format(subject)
    # 读取特征数据和模型分类结果数据
    feature_files = os.listdir(features_path)
    for filename  in feature_files:
        if 'fold{}_'.format(kfold) in filename:
            with open(features_path + filename, "rb") as attnData:
                data = pickle.load(attnData)
                data = np.reshape(data, (-1, 8, data.shape[1], data.shape[2]))

    print('Data loading complete!')

    for head in range(data.shape[1]):
        for i in range(data.shape[0]//8):
            plt.imshow(data[i][head], cmap=plt.cm.Reds, vmax=1., vmin=-1.)
            plt.colorbar()
            plt.tight_layout()
            output_dir_ = output_dir +  "head_{:}/".format(head)
            if not os.path.exists(output_dir_):
                os.makedirs(output_dir_)
            output_attn = output_dir_ + "kfold{:}_head{:}_{:}.png".format(kfold, head, i)
            plt.savefig(output_attn, dpi=dpi)
            print(f'The picture is saved successfully!\nSave address: ' + output_attn)
            plt.clf()



### 模型可解释性——可视化处理——一键生成MSA的注意力信息效果图
def Attention_Weight_Visualization_bef(attention_path, subject, kfold, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
        output_dir: The address where the picture is saved
    '''
    plt.rc('font',family='Times New Roman') # 设置画布的全局字体
    # 图片输出路径
    output_dir = output_dir + 's{:}/'.format(subject) + 'attention figure/kFold_{:}/'.format(kfold)
    # 获取所有与特征相关的数据的地址，并将其保存在一个列表
    features_path = attention_path + 's{:}/'.format(subject)
    # 读取特征数据和模型分类结果数据
    feature_files = os.listdir(features_path)
    for filename  in feature_files:
        if 'fold{}_'.format(kfold) in filename:
            with open(features_path + filename, "rb") as attnData:
                data = pickle.load(attnData)
                data = np.reshape(data, (-1, 8, data.shape[1], data.shape[2]))
    print('Data loading complete!')

    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(18, 2))
    samples_idx = np.random.randint(data.shape[0], size=8)

    for head in range(data.shape[1]): # 列

        im = axs[head].imshow(data[samples_idx[head]][head], cmap=plt.cm.Reds, vmax=1., vmin=-1.)
        axs[head].set_title("head {:}".format(head+1), fontdict={'size':16, 'weight':'bold'})

        # if head == 7:
        #     fig.colorbar(im, ax=axs[head])

    # plt.show()
    output_dir_cycleValue = output_dir + "kfold{:}.png".format(kfold)
    plt.savefig(output_dir_cycleValue, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: ' + output_dir_cycleValue)



### 模型可解释性——可视化处理——将时间卷积层的卷积核权重系数用脑电地形图的方式展现出来
def convWeight_to_waveform(model_path, kfold, subject, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the temporal convolution layer in the form of EEG waveform.

    Args:
        model_path: the path of the model
    '''
    print('Loading model parameters……')
    # 加载模型参数
    model_files = os.listdir(model_path)
    for filename  in model_files:
        if 'fold{}_'.format(kfold) in filename:
            file_name = filename
            file_path = os.path.join(model_path, file_name)
            state_dict = torch.load(file_path)
    print('The model parameters are loaded!')
    print('Image is being generated……')
    # 读取时间卷积层的卷积核权重
    temporalConv_wieght = state_dict['sincConv.weight'].cpu().numpy()
    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(nrows=4, ncols=8, figsize=(24, 12))
    # 在画布上绘制32个地形图
    for i in range(32):
        axs[i//8, i%8].plot(temporalConv_wieght[i,0,0,:])
    # axs[i//8, i%8].set_xlim(0,0.125)
        axs[i//8, i%8].set_ylim(-0.2,0.2)
    output_dir = output_dir + 's{:}/'.format(subject)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = output_dir + "Weight_waveform.png"
    plt.savefig(output_dir, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: '+output_dir)
    plt.show()



### 模型可解释性——可视化处理——将深度卷积层的卷积核权重系数用脑电地形图的方式展现出来
def convWeight_to_topography(model_path, kfold, montage_path, subject, output_dir, dpi=300):
    '''
    Model Interpretability—Visualization—Display the convolution kernel weight coefficients of the deep convolution layer in the form of EEG topography.

    Args:
        model_path: the path of the model
        montage_path: the path of the montage
    '''
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
    # 加载自己修改的（好看的）脑电通道地形图文件
    data1020 = pd.read_excel(montage_path, index_col=0)
    channels1020 = np.array(data1020.index)
    value1020 = np.array(data1020)
    list_dic = dict(zip(channels1020, value1020))
    montage_1020 = mne.channels.make_dig_montage(ch_pos=list_dic,
                                                 nasion=[5.27205792e-18,  8.60992398e-02, -4.01487349e-02],
                                                 lpa=[-0.08609924, -0., -0.04014873],
                                                 rpa=[0.08609924,  0., -0.04014873])
    # 读取深度卷积层的卷积核权重
    depthConv_wieght = state_dict['conv_depth.weight'].cpu().numpy()
    shape = depthConv_wieght.shape
    for i in range(0,shape[0]):
        if i == 0:
            data = depthConv_wieght[i][0]
        else:
            data = np.concatenate((data, depthConv_wieght[i,0]), axis=1)
    # 创建info对象
    info = mne.create_info(ch_names=montage_1020.ch_names, sfreq=1., ch_types='eeg')
    # 设置 EEG/sEEG/ECoG/DBS/fNIRS 通道位置和数字化点
    info.set_montage(montage_1020)
    # 设置画布行列数和画布的尺寸
    fig, ax = plt.subplots(8, 8, figsize=(32, 16))
    # 在画布上绘制32个地形图
    for i in range(0,64):
        im, cn = mne.viz.plot_topomap(data[:, i], info, show=False, axes=ax[i%2+(i//16)*2,(i%16)//2], extrapolate='local')
        # ax[i%4,i//4].set(title="Model coefficients\nbetween delays  and ")
    # 添加颜色棒
    fig.colorbar(im, ax=ax.ravel().tolist())
    output_dir = output_dir + 's{:}/'.format(subject)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = output_dir + "Weight_topo.png"
    plt.savefig(output_dir, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: '+output_dir)
    plt.show()



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



# 模型可解释性——可视化处理——计算被试的CSP特征并把CSP的patterns参数以地形图的形式绘制出来
def FBCSP_topography(data_path, fs, window_details, montage_path, n_components, subject, output_dir, dpi=300):
    '''
    Model interpretability - visualization processing - calculate the CSP characteristics of the subjects and draw the CSP patterns parameters in the form of topographic maps.

    Args:
        data_path: The address of the data being tested
        fs: Data sampling frequency
        window_details: Window for intercepting data
        montage_path: The address of EEG topography
        n_components: The number of components of CSP
        subject: Subject
    '''
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
    # 创建CSP对象
    csp = CSP(n_components, reg=None, log=True, norm_trace=False)
    # 设置画布行列数和画布的尺寸
    fig, ax = plt.subplots(4, 15, figsize=(40, 8))
    print('Image is being generated……')
    for fb in range(0, shape[0]):
        data = filtered_data[fb]
        csp.fit(data, labels)
        filters = csp.filters_.T
        for n_component in range(0, n_components):
            im, cn = mne.viz.plot_topomap(filters[:, n_component], info, show=False, axes=ax[n_component, fb], extrapolate='head')
    fig.colorbar(im, ax=ax.ravel().tolist())
    output_dir = output_dir + 's{:}/'.format(subject)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = output_dir + "FBCSP_topo.png"
    plt.savefig(output_dir, dpi=dpi) 
    print(f'The picture is saved successfully!\nSave address: '+output_dir)
    plt.show()



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
    # filters = [[12, 15, 23, 25, 4, 20, 31, 8],
    #            [8,  18, 19, 21, 2, 11, 15, 31], 
    #            [18, 31, 32, 25, 7, 16,  6, 4]]
    filters = [[12, 15, 25, 8],
               [8,  21, 11, 31], 
               [18, 25, 16, 6]]
    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
    # 对三个卷积的权重进行循环绘图
    for conv in range(3):
        time_end = 0.125*(2**conv)
        num_point = 4*(2**conv)
        # 在画布上绘制8个地形图
        for i in range(4):
            axs[conv, i].plot(np.linspace(0,time_end,num_point,endpoint=True), temporalConv_wieght[conv][(filters[conv][i]-1)//4,(filters[conv][i]-1)%4,0,:])
            if conv == 0:
                axs[conv, i].set_title("Filter {}".format(i+1), fontdict={'size':16, 'weight':'bold'}) # 
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
    output_dir = output_dir + 's{:}/'.format(subject) + 'Inception Weight/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = output_dir + 'Inception Weight Visualization kfold_{:}_4.png'.format(kfold)
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
    data, labels, _, _, _, _ = get_data(data_path, subject, LOSO=False, data_model='two_session', data_type='2a')
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




#%%
###============================ Initialization parameters ============================###
subject = 8
kfold = 1

data_path = "/home/pytorch/LiangXiaohan/MI_Dataverse/BCICIV_2a/mat/"
model_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Ablation study/EEGInc cnnMSA TCN FM/"
montage_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/1020-22.xlsx"
output_dir = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/EEGInc cnnMSA TCN FM/weight visualization/"
# feature_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/t-sne/feature/"
# decision_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/t-sne/decision/"
feature_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/EEGInc cnnMSA TCN FM/t-sne/feature/"
decision_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/EEGInc cnnMSA TCN FM/t-sne/decision/"
output_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/EEGInc cnnMSA TCN FM/t-sne/output/"
attention_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/EEGInc cnnMSA TCN FM/attention/"
# attention_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/ATCNet/Visualization/attention/"
attention_path_bef = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/attention/"
inception_weight_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-dependent/MyModel/Visualization/EEGInc cnnMSA TCN FM/weight visualization/"

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

    # plot_tsne_feature(data_path, feature_path, subject=7, kfoldNum=5, output_dir=feature_path, dpi=600, constitute=False)

    # plot_tsne_decision(data_path, decision_path, subject=7, kfoldNum=5, output_dir=decision_path, dpi=600, constitute=False)

    # plot_tsne_output(data_path, output_path, subject=1, kfoldNum=5, output_dir=output_path, dpi=600, constitute=False)

    # inceptionWeight_to_waveform(model_path, subject=7, kfoldNum=5, output_dir=output_dir, dpi=600)

    # Inception_Weight_Visualization(model_path, subject=1, kfold=3, output_dir=inception_weight_path, dpi=600)

    # attention_weight_visualization(attention_path, subject=7, kfold=0, output_dir=attention_path, dpi=100)

    Attention_Weight_Visualization(attention_path, subject=7, kfold=2, output_dir=attention_path, dpi=600)

    # attention_weight_visualization_bef(attention_path_bef, subject=7, kfold=0, output_dir=attention_path_bef, dpi=100)

    # Attention_Weight_Visualization_bef(attention_path_bef, subject=1, kfold=0, output_dir=attention_path_bef, dpi=600)

    # Attention_Weight_Visualization_hybrid(attention_path_bef, subject=7, kfold=0, output_dir=attention_path_bef, dpi=600)

    # convWeight_to_waveform(model_path, kfold, subject, output_dir, dpi=300)

    # convWeight_to_topography(model_path, kfold, montage_path, subject, output_dir, dpi=300)

    # FBCSP_topography(data_path, fs, window_details, montage_path, n_components, subject, output_dir, dpi=300)

    # weight_FBCSP_pearson(model_path, kfold, data_path, fs, window_details, n_components, output_dir, subject)

    # tempDepth_Weight_Visualization(model_path, montage_path, subject, kfold, output_dir, dpi=600)

    # FBCSP_Weight_Visualization(data_path, model_path, subject, kfold, output_dir, dpi=600)

    # confusion_matrix_disply(confusion_2a, labels_2a, output_dir_cm, dataset='2a', dpi=600)

    # confusion_matrix_disply(confusion_2b, labels_2b, output_dir_cm, dataset='2b', dpi=600)


if __name__ == "__main__":
    main()
