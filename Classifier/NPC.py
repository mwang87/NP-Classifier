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
  
import pickle
with open('Important_DB/Final_Dataset/char2idx_super.pkl','rb') as char2idx_super:
    super_class = pickle.load(char2idx_super)
with open('Important_DB/Final_Dataset/char2idx_sub.pkl','rb') as char2idx_sub:
    sub_class = pickle.load(char2idx_sub)
with open('Important_DB/Final_Dataset/char2idx_sub_terpene.pkl','rb') as char2idx_sub_terpene:
    sub_class_terpene = pickle.load(char2idx_sub_terpene)
with open('Important_DB/Final_Dataset/0507_DNP_class_subclass_set.pkl','rb') as classess:
    classes = pickle.load(classess)
with open('Important_DB/Final_Dataset/0507_DNP_class_subclass_set_terpene.pkl','rb') as classess_terpene:
    classes_terpene = pickle.load(classess_terpene)

# This part may take memory leakage
Super_model = keras.models.load_model('Final_model/FP to Classifier/(0426)DNP_based_super_classifier_final.hdf5')
Lignans_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Lignans.hdf5')
Benzofuranoids_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Benzofuranoids.hdf5')
Steroids_and_Sterols_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Steroids and Sterols.hdf5')
Aminoacids_and_peptides_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Aminoacids and peptides.hdf5')
Flavonoids_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Flavonoids.hdf5')
Lipids_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Lipids.hdf5')
Oxygenheterocycles_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Oxygenheterocycles.hdf5')
Benzopyranoids_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Benzopyranoids.hdf5')
Alkaloids_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Alkaloids.hdf5')
Tannins_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Tannins.hdf5')
Simple_aromatic_natural_products_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Simple aromatic natural products.hdf5')
Polyketides_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Polyketides.hdf5')
Polypyrroles_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Polypyrroles.hdf5')
Carbohydrates_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Carbohydrates.hdf5')
Polypyrroles_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Polypyrroles.hdf5')
Terpenoids_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Terpenoids.hdf5')
Polycyclic_aromatic_natural_products_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Polycyclic aromatic natural products.hdf5')
Triterpenoids_model = keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Terpenoids_Triterpenoids.hdf5')
Sesquiterpenoids_model= keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Terpenoids_Sesquiterpenoids.hdf5')
Diterpenoids_model= keras.models.load_model('Final_model/FP to Classifier/Light_0507_DNP_based_sub_classifier_Terpenoids_Diterpenoids.hdf5')

def classifier(smiles):
    super_list = []
    sub_list = []
    terpene_sub_list = []
    try:
        
        fp = HSQC_logical_r(smiles,2)[0]
        n = np.where(Super_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
        
            
        for j in n:
            super_list.append([name for name, age in super_class.items() if age == j][0])

        for sup in super_list:
            if sup == 'Lignans':
                subs = np.where(Lignans_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Lignans'][m])
                
            elif sup == 'Benzofuranoids':
                subs = np.where(Benzofuranoids_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Benzofuranoids'][m])
                 
            elif sup == 'Steroids and Sterols':
                subs = np.where(Steroids_and_Sterols_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Steroids and Sterols'][m])
                
            elif sup == 'Aminoacids and peptides':
                subs = np.where(Aminoacids_and_peptides_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Aminoacids and peptides'][m])
                
            elif sup == 'Flavonoids':
                subs = np.where(Flavonoids_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Flavonoids'][m])
                 
            elif sup == 'Lipids':
                subs = np.where(Lipids_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Lipids'][m])
                
            elif sup == 'Oxygenheterocycles':
                subs = np.where(Oxygenheterocycles_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Oxygenheterocycles'][m])
                
            elif sup == 'Benzopyranoids':
                subs = np.where(Benzopyranoids_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Benzopyranoids'][m])
                
            elif sup == 'Alkaloids':
                subs = np.where(Alkaloids_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Alkaloids'][m])
             
            elif sup == 'Tannins':
                subs = np.where(Tannins_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Tannins'][m])
                
            elif sup == 'Simple aromatic natural products':
                subs = np.where(Simple_aromatic_natural_products_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Simple aromatic natural products'][m])
                 
            elif sup == 'Polyketides':
                subs = np.where(Polyketides_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Polyketides'][m])
            
            elif sup == 'Polypyrroles':
                subs = np.where(Polypyrroles_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Polypyrroles'][m])
            
            elif sup == 'Carbohydrates':
                subs = np.where(Carbohydrates_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Carbohydrates'][m])
            
            elif sup == 'Terpenoids':
                subs = np.where(Terpenoids_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Terpenoids'][m])
                for n in sub_list:
                    if n == 'Triterpenoids':
                        subs_terpene = np.where(Triterpenoids_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                        for o in subs_terpene:
                            terpene_sub_list.append(classes_terpene['Triterpenoids'][o])
                    elif n == 'Diterpenoids':
                        subs_terpene = np.where(Diterpenoids_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                        for o in subs_terpene:
                            terpene_sub_list.append(classes_terpene['Diterpenoids'][o])
                    elif n == 'Sesquiterpenoids':
                        subs_terpene = np.where(Sesquiterpenoids_model.predict(fp.reshape(1,6144))[0]>=0.5)[0]
                        for o in subs_terpene:
                            terpene_sub_list.append(classes_terpene['Sesquiterpenoids'][o])
            elif sup == 'Polycyclic aromatic natural products':
                subs = np.where(Polycyclic_aromatic_natural_products_model.predict(fp.reshape(1,6144))[0]>0.5)[0]
                for m in subs:
                    sub_list.append(classes['Polycyclic aromatic natural products'][m])
                
    except:
        return print('Error')
    
    return super_list,sub_list, terpene_sub_list
