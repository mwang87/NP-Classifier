#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Module loading
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
import pandas as pd
K = tf.keras.backend
import datetime
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

pd.options.display.max_rows = 15

from tqdm import tqdm

from math import sqrt
import copy


from tqdm import tqdm

# load validation set
val_list = os.listdir('G:/validation_set/X')
X_val = np.zeros((len(val_list),200,240,2),int)
Y_val = np.zeros((len(val_list),2048),int)
for i in range(len(val_list)):
    a = np.load('G:/validation_set/X/{}'.format(val_list[i]))
    #a[0,:,:] =1
    #a[:,0,:] = 1
    #a[:,239,:] = 1
    #a[199,:,:] =1
    X_val[i] = a
    Y_val[i] = np.load('G:/validation_set/Y_QC/{}'.format(val_list[i]))

print('Loading data is done')


# In[2]:


class TrainGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(200,240), n_channels=2,
                 n_classes=40, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X,y = self.__data_generation(list_IDs_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,2048), dtype=float)

        # Generate data
        
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # shift
            train = np.load('G:/training_set/X/{}'.format(ID))
            origin_0 = train[:,:,0]
            origin_0_C = np.where(origin_0==1)[0]
            origin_0_H = np.where(origin_0==1)[1]

            X_ID = np.zeros((200,240,2), dtype=int)

            try:
                for j in range(int(np.sum(origin_0))):
                    Cmove = np.random.randint(-1,2)
                    Hmove = np.random.randint(-1,2)
                    X_ID[origin_0_C[j]+Cmove][origin_0_H[j]+Hmove][0] = 1
            except:
                X_ID[:,:,0] = origin_0


            origin_1 = train[:,:,1]
            origin_1_C = np.where(origin_1==1)[0]
            origin_1_H = np.where(origin_1==1)[1]

            for k in set(origin_1_C):
                para_Cmove = np.random.randint(-1,2)
                try:
                    for l in np.where(origin_1_C==k)[0]:
                        X_ID[k+para_Cmove][origin_1_H[l]+np.random.randint(-1,2)][1] = 1
                except:
                    X_ID[:,:,1] = origin_1
            #X_ID[0,:,:] =1
            #X_ID[:,0,:] = 1
            #X_ID[:,239,:] = 1
            #X_ID[199,:,:] =1
            X[i] = X_ID
            
            # Store class
            y[i] = np.load('G:/training_set/Y_QC/{}'.format(ID))

        return X, y



# In[3]:


#multi gpu
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
log_dir="logs\(011020)HWK_sAug_0102_2ch_QC"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoints/(011020)HWK_sAug_0102_2ch_QC.hdf5',monitor='val_loss',mode='min',save_best_only=True)


# In[4]:.


params = {'dim': (200,240),
          'batch_size': 16,
          'n_classes': 40,
          'n_channels': 2,
          'shuffle': True}

train_list = os.listdir('G:/training_set/X')

training_generator = TrainGenerator(train_list, **params)


model = keras.models.load_model('checkpoints/(011020)HWK_sAug_0102_2ch_0.2.hdf5')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),loss='binary_crossentropy',metrics=['cosine_proximity'])

model.fit_generator(training_generator,
            validation_data=(X_val,Y_val),
                callbacks=[checkpoint,tensorboard_callback],epochs=300,
            use_multiprocessing=False, verbose=1)





