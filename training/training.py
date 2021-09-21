import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import Augmentation
import Model
from fingerprint_handler import _isglycoside
from fingerprint_handler import calculate_fingerprint
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"


np.random.seed(0)


with open('Data/char2idx_class_V1.pkl','rb') as f:
    class_  = pickle.load(f)
with open('Data/char2idx_super_V1.pkl','rb') as f:
    superclass_  = pickle.load(f)
with open('Data/char2idx_path_V1.pkl','rb') as f:
    pathway_  = pickle.load(f)

with open('Data/datset_class_all_V1.pkl','rb') as r:
    dataset = pickle.load(r)
    
def data_generation(idx,data):
    X_train_f = np.zeros((len(idx),2048),int)
    X_train_b = np.zeros((len(idx),4096),int)
    Y_train_path = np.zeros((len(idx),len(pathway_)),int)
    Y_train_super = np.zeros((len(idx),len(superclass_)),int)
    Y_train_class = np.zeros((len(idx),len(class_)),int)
    for i,n in enumerate(idx):
        smiles = data[n]['SMILES']
        X_train_f[i] = calculate_fingerprint(smiles,2)[0]
        X_train_b[i] = calculate_fingerprint(smiles,2)[1]
        #Y_train_path[i] = dataset[n]['Pathway']
        #Y_train_super[i] = dataset[n]['Super_class']
        Y_train_class[i] = data[n]['Class']
    return [X_train_f,X_train_b], Y_train_class

def top_k_categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)
    
# Train, Validation, and test set 

b_key = list(dataset.keys())
np.random.shuffle(b_key)
dict_ = np.array(b_key)
Y_ = np.array([ np.max(np.where(dataset[i]['Class']==1)[0]) for i in dict_])

train_D, test_dict, y_train, y_test = train_test_split(dict_, Y_, test_size=0.2, random_state=1, stratify = Y_)
train_dict, val_dict, y_train, y_val = train_test_split(train_D, y_train, test_size=0.2, random_state=1, stratify = y_train)

#Implement data augmentation
aug = {}
for i in tqdm(train_dict):
    smiles = dataset[i]['SMILES']
    ori_path = dataset[i]['Pathway']
    ori_sup = dataset[i]['Super_class']
    ori_class = dataset[i]['Class']
    if _isglycoside(smiles) != True:
        smiles_list = Augmentation.methoxy(smiles)
        for m in smiles_list:
            inchi_key = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(m))[:14]
            if inchi_key not in b_key:
                aug[inchi_key] = {'SMILES':m,'Pathway':ori_path,'Super_class':ori_sup,'Class':ori_class}

        smiles_list = Augmentation.demethoxy(smiles)
        for m in smiles_list:
            inchi_key = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(m))[:14]
            if inchi_key not in b_key:
                aug[inchi_key] = {'SMILES':m,'Pathway':ori_path,'Super_class':ori_sup,'Class':ori_class}
        aug[i] = {'SMILES':smiles,'Pathway':ori_path,'Super_class':ori_sup,'Class':ori_class}

    else:
        inchi_key = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]
        aug[i] = {'SMILES':smiles,'Pathway':ori_path,'Super_class':ori_sup,'Class':ori_class}        

X_train, Y_train = data_generation(list(aug.keys()),aug)
X_val, Y_val = data_generation(val_dict,dataset)

model = Model()

#Avoiding overfitting
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model.fit(x=X_train, y=Y_train, batch_size=128, epochs=100, verbose=2,callbacks=[early_stop], validation_split=0.0, validation_data=(X_val,Y_val), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
