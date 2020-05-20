from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from tqdm import tqdm
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

import tensorflow as tf
from tensorflow import keras

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #for cpu environment

def isNaN(num):
    return num != num

import json
#Loading dictionary of super_class, class and subclass(terpene)

def main():
    with open('dict/NPC.json','r') as total:
        class_ = json.load(total)


    # This part may take memory leakage
    #Super_class
    model = keras.models.load_model('model/(0519)DNP_based_Allinone_final.hdf5')

def classifier(smiles):
    result=[]
    try: 
        fp = calculate_fingerprint(smiles,2)
        pred = model.predict(fp)[0]
        n = np.where(pred>=0.5)[0]
        for i in n:
            result.append(class_[str(i)])
    except: # if there are no elements with over 0.5, then element with maximum value will be suggested
        pass
    if result == []:
        n = np.argmax(pred)
        
        result = class_[str(n)]
        
    
    return result
    


