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
import copy
import torch
import numpy as np
from dataLoad.preprocess import cross_validate

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

from torch.utils.data import DataLoader

from visdom import Visdom
import datetime



#%% Model parameter reset
def reset_parameters(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


#%% Model validation functions
def validate_model(model, dataset, device, losser, batch_size=128, n_calsses=4):
    loader = DataLoader(dataset, batch_size=batch_size)
    loss_val = 0.0
    accuracy_val = 0.0
    confusion_val = np.zeros((n_calsses,n_calsses), dtype=np.int8)
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            target = target.to(device)

            probs = model(inputs)
            loss = losser(probs, target)

            loss_val += loss.detach().item()
            accuracy_val += torch.sum(torch.argmax(probs,dim=1) == target, dtype=torch.float32)

            y_true = target.to('cpu').numpy()
            y_pred = probs.argmax(dim=-1).to('cpu').numpy()
            confusion_val += confusion_matrix(y_true, y_pred)
        
        loss_val = loss_val / len(loader)
        accuracy_val = accuracy_val / len(dataset)

    return loss_val, accuracy_val, confusion_val


#%% Improved two-stage model training strategy
def train_with_cross_validate(model_name, subject, frist_epochs, eary_stop_epoch, second_epochs, kfolds, batch_size, 
                              device, X_train, Y_train, model, losser, model_savePath, n_calsses):
    '''
    The function of the model train with cross validate.

    Args:
        model_name: Model being trained
        subject: Trained subject
        frist_epochs: The number of epochs in the first stage
        eary_stop_epoch: The number of epochs for early stopping
        second_epochs: The number of epochs in the second stage
        kfolds: The number of folds 
        batch_size: Batch size
        device: Device for model training
        X_train: The train data
        Y_train: The train label
        model_savePath: Path to save the model
        n_calsses: Number of categories

    '''
    # vis = Visdom(env='main')  # 设置环境窗口的名称,如果不设置名称就默认为main
    # opt_train_acc = {'xlabel':'epochs', 'ylabel':'acc_value', 'title':model_name+'_train_acc'}
    # opt_eval_acc = {'xlabel':'epochs', 'ylabel':'acc_value', 'title':model_name+'_eval_loss'}
    # opt_lr = {'xlabel':'epochs', 'ylabel':'acc_value', 'title':model_name+'_eval_acc'}

    # Create a file to store performance during training
    log_write = open(model_savePath + "/log.txt", "w")
    log_write.write( '\nTraining on subject '+ str(subject) +'\n')

    best_acc_list = []
    avg_eval_acc = 0
    for kfold, (train_dataset,valid_dataset) in enumerate(cross_validate(X_train, Y_train, kfolds)):
        # train_acc_window = vis.line(X=[0], Y=[0.3], opts=opt_train_acc)
        # eval_acc_window = vis.line(X=[0], Y=[0.3], opts=opt_eval_acc)
        # lr_window = vis.line(X=[0], Y=[0], opts=opt_lr)

        info = 'Subject_{}_fold_{}:'.format(subject, kfold)
        print(info)
        log_write.write(info +'\n')

        print(len(train_dataset),len(valid_dataset))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.apply(reset_parameters)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3) # , weight_decay=1e-3
        scheduler = None
        scheduler_adap = None

        ### First step
        best_loss_kfold = np.inf
        best_loss_kfold_acc = 0
        best_acc_kfold = 0
        best_acc_kfold_loss = np.inf
        mini_loss = None
        remaining_epoch = eary_stop_epoch
        for iter in range(frist_epochs):
            loss_train = 0
            accuracy_train = 0

            model.train()
            for inputs, target in train_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)

                optimizer.zero_grad() # 清空梯度
                output = model(inputs) # 前向传播和计算损失
                loss = losser(output, target)
                loss.backward() # 反向传播和计算梯度
                optimizer.step() # 更新参数

                accuracy_train += torch.sum(torch.argmax(output,dim=1) == target, dtype=torch.float32) / len(train_dataset)
                loss_train += loss.detach().item() / len(train_dataloader)

            loss_val, accuracy_val, confusion_val = validate_model(model, valid_dataset, device, losser, n_calsses=n_calsses)

            remaining_epoch = remaining_epoch-1

            if scheduler:
                if scheduler_adap:
                    current_lr = optimizer.state_dict()['param_groups'][0]['lr']  # 当前学习率
                    scheduler.step(accuracy_val)  # 调整学习率
                else:
                    current_lr = scheduler.get_last_lr()[0]
                    scheduler.step()  # 调整学习率
            else:
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']  # 当前学习率

            # vis.line(X=[iter], Y=[accuracy_train.cpu()], win=train_acc_window, opts=opt_train_acc, update='append')
            # vis.line(X=[iter], Y=[loss_val], win=eval_acc_window, opts=opt_eval_acc, update='append')
            # vis.line(X=[iter], Y=[accuracy_val.cpu()], win=lr_window, opts=opt_lr, update='append')

            if remaining_epoch <=0:
                avg_eval_acc += best_acc_kfold
                break
            if  mini_loss is None or loss_train<mini_loss:
                mini_loss = loss_train

            if loss_val < best_loss_kfold:
                if accuracy_val >= best_acc_kfold:
                    best_model = copy.deepcopy(model.state_dict())
                    optimizer_state = copy.deepcopy(optimizer.state_dict())
                    best_acc_kfold = accuracy_val
                    best_acc_kfold_loss = loss_val
                remaining_epoch = eary_stop_epoch
                best_loss_kfold = loss_val
                best_loss_kfold_acc = accuracy_val

            if accuracy_val > best_acc_kfold:
                best_model = copy.deepcopy(model.state_dict())
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_acc_kfold = accuracy_val
                best_acc_kfold_loss = loss_val
                remaining_epoch = eary_stop_epoch

            info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = info + '\tKfold:{0:1}\tEpoch:{1:3}\tTra_Loss:{2:.3}\tTr_acc:{3:.3}\tVa_Loss:{4:.3}\tVa_acc:{5:.3}\tMinVloss:{6:.3}\tToacc:{7:.3}\tMaxVacc:{8:.3}\tToloss:{9:.3}\tramainingEpoch:{10:3}'\
                   .format(kfold+1, iter, loss_train, accuracy_train, loss_val, accuracy_val, best_loss_kfold, best_loss_kfold_acc, best_acc_kfold, best_acc_kfold_loss, remaining_epoch)
            # info = info + '\tKfold:{0:1}\tEpoch:{1:3}\tTra_Loss:{2:.3}\tTr_acc:{3:.3}\tVa_Loss:{4:.3}\tVa_acc:{5:.3}\tMaxVacc:{6:.3}\tToloss:{7:.3}\tramainingEpoch:{8:3}'\
            #        .format(kfold+1, iter, loss_train, accuracy_train, loss_val, accuracy_val, best_acc_kfold, best_acc_kfold_loss, remaining_epoch)
            print(info)
            # print(f'confusion:\n{confusion_val}\n')
            log_write.write(info +'\n')

        info = f'Earyly stopping at Epoch {iter},and retrain the Net using both the training data and validation data.'
        print(info)
        log_write.write(info +'\n')

        ### Second step
        model.load_state_dict(best_model)
        optimizer.load_state_dict(optimizer_state)

        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        for iter in range(second_epochs):
            model.train()
            for inputs, target in train_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad() # 清空梯度
                output = model(inputs) # 前向传播和计算损失
                loss = losser(output, target)
                loss.backward() # 反向传播和计算梯度
                optimizer.step() # 更新参数

            for inputs, target in valid_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad() # 清空梯度
                output = model(inputs) # 前向传播和计算损失
                loss = losser(output, target)
                loss.backward() # 反向传播和计算梯度
                optimizer.step() # 更新参数

            loss_val, accuracy_val, confusion_val = validate_model(model, valid_dataset, device, losser, n_calsses=n_calsses)

            info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = info + '\tKfold:{0:1}\tEpoch:{1:3}\tVa_Loss:{2:.3}\tVa_acc:{3:.3}'.format(kfold+1, iter, loss_val, accuracy_val)
            print(info)
            # print(f'confusion:\n{confusion_val}\n')
            log_write.write(info +'\n')

            if loss_val < mini_loss:
                break

        file_name = '{}_sub{}_fold{}_acc{:.4}.pth'.format(model_name, subject, kfold, best_acc_kfold)
        print(file_name)
        torch.save(model.state_dict(), os.path.join(model_savePath, file_name))
        # torch.save(best_model, os.path.join(model_savePath, file_name))

        info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info = info + 'The model was saved successfully!'
        print(info)
        log_write.write(info +'\n')

    info = f"Avg_eval_Acc : {avg_eval_acc*100/kfolds:4f}"
    print(info)
    log_write.write(info +'\n')
    log_write.close()

