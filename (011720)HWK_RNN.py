#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
pd.options.display.max_rows = 15
from tqdm import tqdm
from math import sqrt
import copy
from keras.preprocessing.sequence import pad_sequences
smiles_element = sorted(set(['#', '%', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'S', '[', ']', 'c', 'l', 'n', 'o', 'r', 's','<','>']))
char2idx = {u:i+1 for i, u in enumerate(smiles_element)}
idx2char = np.array(smiles_element)
vocab_size = len(smiles_element)+1
from IPython.display import SVG
from keras.utils.vis_utils import plot_model
from keras.utils import model_to_dot
from keras.utils import to_categorical


# In[3]:


def create_sequences(max_length, caption, image):
    # X1 : input for image features
    # X2 : input for text features
    # y  : output word
    X1, X2, y = list(), list(), list()
    vocab_size = 36
    # Walk through each caption for the image
    
    # Split one sequence into multiple X,y pairs
    for i in range(1, len(caption)):
        # Split into input and output pair
        in_seq, out_seq = caption[:i], caption[i]
        # Pad input sequence
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        # Encode output sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # Store
        X1.append(image)
        X2.append(in_seq)
        y.append(out_seq)
    return X1, X2, y

def train_generator(train_list, max_length, batch_size):
    # Setting random seed for reproducibility of results
    # Image ids
    image_ids = train_list
    _count=0
    assert batch_size<= len(image_ids), 'Batch size must be less than or equal to {}'.format(len(image_ids))
    while True:
        if _count >= len(image_ids):
            # Generator exceeded or reached the end so restart it
            _count = 0
        # Batch list to store data
        input_img_batch, input_sequence_batch, output_word_batch = list(), list(), list()
        for i in range(_count, min(len(image_ids), _count+batch_size)):
            # Retrieve the image id
            image_id = image_ids[i]
            # Retrieve the image features
            image = np.load('G:/training_set/X_embd_2ch/{}'.format(image_id))

            # Retrieve the captions list
            caption = np.load('G:/training_set/smiles_100/{}'.format(image_id))
            # Shuffle captions list
            
            input_img, input_sequence, output_word = create_sequences(max_length, caption, image)
            # Add to batch
            for j in range(len(input_img)):
                input_img_batch.append(input_img[j])
                input_sequence_batch.append(input_sequence[j])
                output_word_batch.append(output_word[j])
        _count = _count + batch_size
        yield [np.array(input_img_batch), np.array(input_sequence_batch)], np.array(output_word_batch)
        
def validation_generator(validation_list, max_length, batch_size):
    # Setting random seed for reproducibility of results
    # Image ids
    image_ids = validation_list
    _count=0
    assert batch_size<= len(image_ids), 'Batch size must be less than or equal to {}'.format(len(image_ids))
    while True:
        if _count >= len(image_ids):
            # Generator exceeded or reached the end so restart it
            _count = 0
        # Batch list to store data
        input_img_batch, input_sequence_batch, output_word_batch = list(), list(), list()
        for i in range(_count, min(len(image_ids), _count+batch_size)):
            # Retrieve the image id
            image_id = image_ids[i]
            # Retrieve the image features
            image = np.load('G:/validation_set/X_embd_2ch/{}'.format(image_id))

            # Retrieve the captions list
            caption = np.load('G:/validation_set/smiles_100/{}'.format(image_id))
            # Shuffle captions list
            
            input_img, input_sequence, output_word = create_sequences(max_length, caption, image)
            # Add to batch
            for j in range(len(input_img)):
                input_img_batch.append(input_img[j])
                input_sequence_batch.append(input_sequence[j])
                output_word_batch.append(output_word[j])
        _count = _count + batch_size

        yield [np.array(input_img_batch), np.array(input_sequence_batch)], np.array(output_word_batch)


# In[4]:


#CNN_image
Input_X1 = layers.Input(shape=(2048,))
X = layers.Dropout(0.2)(Input_X1)
X = layers.Dense(256, activation = 'relu')(X)

#RNN
Input_X2 = layers.Input(shape=(100,))
Y = layers.Embedding(vocab_size, 256, mask_zero=True)(Input_X2)
Y = layers.LSTM(256)(Y)

#decoder
Z = layers.add([X,Y])
Z = layers.Dense(256, activation = 'relu')(Z)
output = layers.Dense(vocab_size,activation='softmax')(Z)
model = keras.Model(inputs=[Input_X1,Input_X2], outputs=output)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[5]:


params = {'max_length': 100,
          'batch_size': 16,}
train_list = os.listdir('G:/training_set/smiles_100')
validation_list = os.listdir('G:/validation_set/smiles_100')
training_generator = train_generator(train_list, **params)
validation_generator = validation_generator(validation_list,**params)


# In[ ]:


model.fit_generator(training_generator,
            epochs=300,
            steps_per_epoch = len(train_list)//16,
            validation_data=validation_generator,
            validation_steps = len(validation_list)//16,
            use_multiprocessing=None,
            verbose=1)


# In[ ]:




