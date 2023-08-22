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

import mne
import numpy as np
import scipy.io as scio



#%%
class Load_BCIC_2a():
    '''
    Subclass of LoadData for loading BCI Competition IV Dataset 2a.

    Methods:
        get_epochs(self, tmin=-0., tmax=2, baseline=None, downsampled=None)
    '''
    def __init__(self, data_path, persion):
        self.stimcodes_train=('769','770','771','772')
        self.stimcodes_test=('783')
        self.data_path = data_path
        self.persion = persion
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(Load_BCIC_2a,self).__init__()

    def get_epochs_train(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        file_to_load = 's{:}/'.format(self.persion) + 'A0' + self.persion + 'T.gdf'
        raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
        if low_freq and high_freq:
            raw_data.filter(l_freq=low_freq, h_freq=high_freq)
        if downsampled is not None:
            raw_data.resample(sfreq=downsampled)
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes_train]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data':self.x_data[:, :, :-1],
                  'y_labels':self.y_labels,
                  'fs':self.fs}
        return eeg_data

    def get_epochs_test(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        file_to_load = 's{:}/'.format(self.persion) + 'A0' + self.persion + 'E.gdf'
        raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
        data_path_label = self.data_path + "true_labels/A0" + self.persion + "E.mat"
        mat_label = scio.loadmat(data_path_label)
        mat_label = mat_label['classlabel'][:,0]-1
        if (low_freq is not None) and (high_freq is not None):
            raw_data.filter(l_freq=low_freq, h_freq=high_freq)
        if downsampled is not None:
            raw_data.resample(sfreq=downsampled)
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes_test]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1]) + mat_label
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data':self.x_data[:, :, :-1],
                  'y_labels':self.y_labels,
                  'fs':self.fs}
        return eeg_data


#%%
class Load_BCIC_2b():
    '''
    Subclass of LoadData for loading BCI Competition IV Dataset 2b.

    Methods:
        get_epochs(self, tmin=-0., tmax=2, baseline=None, downsampled=None)
    '''
    def __init__(self, data_path, persion):
        self.stimcodes_train=('769','770')
        self.stimcodes_test=('783')
        self.data_path = data_path
        self.persion = persion
        self.channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        self.train_name = ['1','2','3']
        self.test_name = ['4','5']
        super(Load_BCIC_2b,self).__init__()

    def get_epochs_train(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        x_data = []
        y_labels = []
        for session in self.train_name:
            file_to_load = 'data/' + 'B0{}0{}T.gdf'.format(self.persion, session)
            raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
            data_path_label = self.data_path + 'label/' + 'B0{}0{}T.mat'.format(self.persion, session)
            mat_label = scio.loadmat(data_path_label)
            mat_label = mat_label['classlabel'][:,0]-1
            if low_freq and high_freq:
                raw_data.filter(l_freq=low_freq, h_freq=high_freq)
            if downsampled is not None:
                raw_data.resample(sfreq=downsampled)
            self.fs = raw_data.info.get('sfreq')
            events, event_ids = mne.events_from_annotations(raw_data)
            stims =[value for key, value in event_ids.items() if key in self.stimcodes_train]
            epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                                baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
            epochs = epochs.drop_channels(self.channels_to_remove)
            self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
            self.x_data = epochs.get_data()
            x_data.extend(self.x_data[:, :, :-1])
            y_labels.extend(self.y_labels)

        x_data = np.array(x_data)
        y_labels = np.array(y_labels)
        eeg_data={'x_data':x_data,
                  'y_labels':y_labels,
                  'fs':self.fs}
        return eeg_data

    def get_epochs_test(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        x_data = []
        y_labels = []
        for session in self.test_name:
            file_to_load = 'data/' + 'B0{}0{}E.gdf'.format(self.persion, session)
            raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
            data_path_label = self.data_path + 'label/' + 'B0{}0{}E.mat'.format(self.persion, session)
            mat_label = scio.loadmat(data_path_label)
            mat_label = mat_label['classlabel'][:,0]-1
            if (low_freq is not None) and (high_freq is not None):
                raw_data.filter(l_freq=low_freq, h_freq=high_freq)
            if downsampled is not None:
                raw_data.resample(sfreq=downsampled)
            self.fs = raw_data.info.get('sfreq')
            events, event_ids = mne.events_from_annotations(raw_data)
            stims =[value for key, value in event_ids.items() if key in self.stimcodes_test]
            epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                                baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
            epochs = epochs.drop_channels(self.channels_to_remove)
            self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1]) + mat_label
            self.x_data = epochs.get_data()
            x_data.extend(self.x_data[:, :, :-1])
            y_labels.extend(self.y_labels)
        
        x_data = np.array(x_data)
        y_labels = np.array(y_labels)
        eeg_data={'x_data':x_data,
                  'y_labels':y_labels,
                  'fs':self.fs}
        return eeg_data


#%%
def load_data_2a(data_path, subject, training, all_trials=True):
    """ Loading and Dividing of the data set based on the subject-specific (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original competition, i.e., 288 x 9 trials in session 1 for training, 
    and 288 x 9 trials in session 2 for testing.  

        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts 
	"""
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6*48 	
    window_Length = 7*250 

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = scio.loadmat(data_path+'A0'+str(subject)+'T.mat')
    else:
        a = scio.loadmat(data_path+'A0'+str(subject)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1     = a_data[0,ii]
        a_data2     = [a_data1[0,0]]
        a_data3     = a_data2[0]
        a_X 		= a_data3[0]
        a_trial 	= a_data3[1]
        a_y 		= a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
            if(a_artifacts[trial] != 0 and not all_trials):
                continue
            data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial +=1

    return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]


#%%
def load_data_LOSO (data_path, subject, train_model='one_session', Transfer=False, trans_num=1): 
    """ Loading and Dividing of the data set based on the 'Leave One Subject Out' (LOSO) evaluation approach. 
    LOSO is used for  Subject-independent evaluation.
   
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
            Here, the subject data is used  test the model and other subjects data
            for training
    """
    
    if train_model == 'one_session':
        X_train, y_train = [], []
        X_train_trans, y_train_trans = [], []
        for sub in range (1,10):
            path = data_path + 's' + str(sub) + '/'

            if (sub == subject):
                X_test, y_test = load_data_2a(path, sub, False)
                # X_test, y_test = load_data_2a(path, sub, True)
                # X, y = load_data_2a(path, sub, False)
                # X_test = np.concatenate((X_test, X), axis=0)
                # y_test = np.concatenate((y_test, y), axis=0)
                if Transfer:
                    X, y = load_data_2a(path, sub, True)
                    class0_idx = np.where(y==1)
                    class1_idx = np.where(y==2)
                    class2_idx = np.where(y==3)
                    class3_idx = np.where(y==4)
                    idx = np.random.randint(class3_idx[0].shape[0], size=trans_num)
                    X_train_trans = X[class0_idx[0][idx]]
                    y_train_trans = y[class0_idx[0][idx]]
                    X_train_trans = np.concatenate((X_train_trans, X[class1_idx[0][idx]]), axis=0)
                    y_train_trans = np.concatenate((y_train_trans, y[class1_idx[0][idx]]), axis=0)
                    X_train_trans = np.concatenate((X_train_trans, X[class2_idx[0][idx]]), axis=0)
                    y_train_trans = np.concatenate((y_train_trans, y[class2_idx[0][idx]]), axis=0)
                    X_train_trans = np.concatenate((X_train_trans, X[class3_idx[0][idx]]), axis=0)
                    y_train_trans = np.concatenate((y_train_trans, y[class3_idx[0][idx]]), axis=0)
            elif (X_train == []):
                X_train, y_train = load_data_2a(path, sub, True)
            else:
                X, y = load_data_2a(path, sub, True)
                X_train = np.concatenate((X_train, X), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)

        return X_train, y_train, X_test, y_test, X_train_trans, y_train_trans

    if train_model == 'two_session':
        X_train, y_train = [], []
        for sub in range (1,10):
            path = data_path + 's' + str(sub) + '/'

            X1, y1 = load_data_2a(path, sub, True)
            X2, y2 = load_data_2a(path, sub, False)
            X = np.concatenate((X1, X2), axis=0)
            y = np.concatenate((y1, y2), axis=0)

            if (sub == subject):
                X_test = X
                y_test = y
            elif (X_train == []):
                X_train = X
                y_train = y
            else:
                X_train = np.concatenate((X_train, X), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)
        
        X_train_trans = []
        y_train_trans = []

        return X_train, y_train, X_test, y_test, X_train_trans, y_train_trans


#%%
def load_data_onLine2a (data_path, data_model='one_session'): 
    """ Loading and Dividing of the data set based on the 'Leave One Subject Out' (LOSO) evaluation approach. 
    LOSO is used for  Subject-independent evaluation.
   
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
            Here, the subject data is used  test the model and other subjects data
            for training
    """
    
    if data_model == 'one_session':
        X_train, y_train = [], []
        for sub in range (1,10):
            path = data_path + 's' + str(sub) + '/'

            if (X_train == []):
                X_train, y_train = load_data_2a(path, sub, True)
            else:
                X, y = load_data_2a(path, sub, True)
                X_train = np.concatenate((X_train, X), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)

        index1 = np.where(y_train == 1)
        index2 = np.where(y_train == 2)

        X_train_2class = np.concatenate((X_train[index1], X_train[index2]), axis=0)
        y_train_2class = np.concatenate((y_train[index1], y_train[index2]), axis=0)

        return X_train_2class, y_train_2class

    if data_model == 'two_session':
        X_train, y_train = [], []
        for sub in range (1,10):
            path = data_path + 's' + str(sub) + '/'

            X1, y1 = load_data_2a(path, sub, True)
            X2, y2 = load_data_2a(path, sub, False)
            X = np.concatenate((X1, X2), axis=0)
            y = np.concatenate((y1, y2), axis=0)

            if (X_train == []):
                X_train = X
                y_train = y
            else:
                X_train = np.concatenate((X_train, X), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)

        index1 = np.where(y_train == 1)
        index2 = np.where(y_train == 2)

        X_train_2class = np.concatenate((X_train[index1], X_train[index2]), axis=0)
        y_train_2class = np.concatenate((y_train[index1], y_train[index2]), axis=0)

        return X_train_2class, y_train_2class
    

#%%
if __name__ == '__main__':
    path = "/home/pytorch/LiangXiaohan/MI_Dataverse/BCICIV_2b/gdf/"
    load_raw_data = Load_BCIC_2b(path, 1)
    eeg_data = load_raw_data.get_epochs_train(tmin=0., tmax=4.)
    print(eeg_data['x_data'].shape, eeg_data['y_labels'].shape)
    eeg_data = load_raw_data.get_epochs_test(tmin=0., tmax=4.)
    print(eeg_data['x_data'].shape, eeg_data['y_labels'].shape)
