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
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

import torch
import numpy as np
import torch.nn as nn
import scipy.io as scio
from sklearn.model_selection import train_test_split

from dataLoad.preprocess import get_data
from functions import train_with_cross_validate, test_with_cross_validate
from functions import train_without_cv_transfer, test_without_cv_transfer
from functions import getModel

from visdom import Visdom
import datetime


print(torch.__version__)

###============================ Use the GPU to train ============================###
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print("current device:",torch.cuda.get_device_name(device))

###============================ Sets the seed for random numbers ============================###
torch.manual_seed(20230520)
if use_cuda:
    torch.cuda.manual_seed(20230520)
np.random.seed(20230520)

###============================ Load data ============================###
###============ BCICIV 2a Database ============###
data_path = "/home/pytorch/LiangXiaohan/MI_Dataverse/BCICIV_2a/mat/"
subject = 1
# X_train, Y_train, X_test, Y_test, _, _ = get_data(data_path, subject, LOSO=False, Transfer=False, trans_num=1, data_type='2a')
# print(X_train.shape, X_test.shape)

X_train, Y_train, X_test, Y_test = None, None, None, None

###============================ Initialization parameters ============================###
###============ Train parameters ============###
frist_epochs    = 3000
eary_stop_epoch = 300
second_epochs   = 800
batch_size      = 64
kfolds          = 5
test_size       = 0.05
eval_num        = None
patience        = 10
threshold       = 0.001
n_calsses       = 4
patience        = 3
threshold       = 0.001
model_name      = 'MyModel'
# model_path      = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-independent/MyModel/one_session/" + 's{:}/'.format(subject)

###============ Initialization model save path ============###
save_path = os.path.join(curPath, 'Saved_files', 'BCIC_2a',  model_name, 's{:}/'.format(subject))
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

###============================ Initialization model ============================###
model = getModel(model_name, device)

model = model.to(device)
print(model)

###============ loss ============###
losser = nn.CrossEntropyLoss().to(device)

###============================ Main function ============================###
def main():
    start = datetime.datetime.now()
    train_with_cross_validate(model_name, subject, frist_epochs, eary_stop_epoch, second_epochs, kfolds, batch_size, 
                              device, X_train, Y_train, model, losser, save_path, n_calsses)
    end = datetime.datetime.now()
    print(f"训练时间：{end-start} \n")

def main_Multisub():
    for subject in range(1,10):
        X_train, Y_train, X_test, Y_test, _, _ = get_data(data_path, subject, LOSO=False, data_model='two_session', data_type='2a')
        print(X_train.shape, X_test.shape)
        save_path = os.path.join(curPath, 'Saved_files', 'BCIC_2a', 'subject-dependent', model_name, 'Training strategy', 'two_stages+acc 3', 's{:}/'.format(subject))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        visualization_path = os.path.join(curPath, 'Saved_files', 'BCIC_2a', 'subject-dependent', model_name, 'Visualization')
        # if not os.path.exists(visualization_path):
        #     os.makedirs(visualization_path)
        start = datetime.datetime.now()
        train_with_cross_validate(model_name, subject, frist_epochs, eary_stop_epoch, second_epochs, kfolds, batch_size, 
                                  device, X_train, Y_train, model, losser, save_path, n_calsses)
        end = datetime.datetime.now()
        print(f"训练时间：{end-start} \n")

def main_Transfer():
    for num in range(5,6):
        for subject in range(2,3):
            model_path = "/home/pytorch/LiangXiaohan/MI_Same_limb/BCICIV_2a/Saved_files/BCIC_2a/subject-independent/MyModel/one_session/" + 's{:}/'.format(subject)
            _, _, X_test, Y_test, X_train, Y_train = get_data(data_path, subject, LOSO=True, data_model='one_session', Transfer=True, trans_num=num, data_type='2a') # trans_num=num, 
            print(X_train.shape, X_test.shape)
            save_path = os.path.join(curPath, 'Saved_files', 'BCIC_2a', 'subject-independent', model_name, 'Transfer Learning', 'Trans num {:}'.format(num), 's{:}-2/'.format(subject))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            start = datetime.datetime.now()
            train_without_cv_transfer(model_name, model_path, subject, kfolds, frist_epochs, eary_stop_epoch, 
                                      batch_size, device, X_train, Y_train, model, losser, save_path)
            end = datetime.datetime.now()
            print(f"训练时间：{end-start} \n")


if __name__ == "__main__":
    # main()
    main_Multisub()
    # main_Transfer()
