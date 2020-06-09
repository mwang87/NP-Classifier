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

#for glycoside check
hexa_pyranose = Chem.MolFromSmarts('[O]C1C([O])C([O])C(C[O])OC1[*]')
penta_furanose = Chem.MolFromSmarts('[O]CC1OC([*])C([O])C1[O]')

import json
#Loading dictionary of pathway, superclass and class

def main():
    with open('dict/NPClass_char2idx_class.json','r') as class_:
        index_class = json.load(class_)
    with open('dict/NPClass_char2idx_super.json','r') as super_:
        index_super = json.load(super_)
    with open('dict/NPClass_char2idx_path.json','r') as path_:
        index_path = json.load(path_)


    # This part may take memory leakage
    #Super_class
    
    model_class = keras.models.load_model('model/NP_classifier_class_final.hdf5')
    model_super = keras.models.load_model('model/NP_classifier_superclass_final.hdf5')
    model_pathway = keras.models.load_model('model/NP_classifier_path_final.hdf5')
    

def classifier(smiles):
    class_result=[]
    superclass_result = []
    pathway_result = []
    
    fp = calculate_fingerprint(smiles,2)
    pred_class = model_class.predict(fp)[0]
    pred_super = model_super.predict(fp)[0]
    pred_path = model_pathway.predict(fp)[0]
    n_class = np.where(pred_class>=0.5)[0]
    n_super = np.where(pred_super>=0.5)[0]
    n_path =np.where(pred_path>=0.5)[0]
    for i in n_class:
        class_result.append(index_class[i])
    if class_result == []:
        k = np.argmax(pred_class)
        class_result.append(index_class[k]+f'({(100*pred_class[k]).round(2)}%)') #if possiblity is under 0.5 
        
    for i in n_super:
        superclass_result.append(index_superclass[i])
    if superclass_result == []:
        k = np.argmax(pred_super)
        superclass_result.append(index_superclass[k]+f'({(100*pred_super[k]).round(2)}%)')
        
    for i in n_path:
        pathway_result.append(index_pathway[i])
    if pathway_result == []:
        k = np.argmax(pred_path)
        pathway_result.append(index_pathway[k]+f'({(100*pred_path[k]).round(2)}%)')
    
    return pathway_result, superclass_result, class_result

def isglycoside(smiles): #now it is expressed as boolean but can be changed to any format
    mol = Chem.MolFromSmiles(smiles)
    if mol.HasSubstructMatch(hexo_pyranose) or mol.HasSubstructMatch(pento_furanose):
        return True 
    else:
        retrun False 
    
