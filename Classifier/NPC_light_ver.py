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
#Loading dictionary of super_class, class and subclass(terpene)

with open('D:/dropbox/ucsd/smart/cnn/ml/Important_DB/Final_Dataset/0512_char2idx_super.pkl','rb') as char2idx_super:
    super_class = pickle.load(char2idx_super)
with open('D:/dropbox/ucsd/smart/cnn/ml/Important_DB/Final_Dataset/0512_char2idx_sub.pkl','rb') as char2idx_sub:
    sub_class = pickle.load(char2idx_sub)
with open('D:/dropbox/ucsd/smart/cnn/ml/Important_DB/Final_Dataset/0512_DNP_class_subclass_set.pkl','rb') as classess:
    classes = pickle.load(classess)
with open('D:/dropbox/ucsd/smart/cnn/ml/Important_DB/Final_Dataset/0513_char2idx_sub_terpene.pkl','rb') as char2idx_sub:
    sub_class_terpene = pickle.load(char2idx_sub)    
with open('D:/dropbox/ucsd/smart/cnn/ml/Important_DB/Final_Dataset/0513_DNP_class_subclass_set_terpene.pkl','rb') as classess:
    classes_terpene = pickle.load(classess)


# This part may take memory leakage
#Super_class
super_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_super_classifier_final_Light.hdf5')
#Class
Lignans_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Lignans_Light.hdf5')
Benzofuranoids_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Benzofuranoids_Light.hdf5')
Steroids_and_Sterols_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Steroids and Sterols_Light.hdf5')
Aminoacids_and_peptides_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Aminoacids and peptides_Light.hdf5')
Flavonoids_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Flavonoids_Light.hdf5')
Aliphatic_natural_products_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Aliphatic natural products_Light.hdf5')
Oxygenheterocycles_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Oxygenheterocycles_Light.hdf5')
Benzopyranoids_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Benzopyranoids_Light.hdf5')
Alkaloids_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Alkaloids_Light.hdf5')
Tannins_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Tannins_Light.hdf5')
Simple_aromatic_natural_products_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Simple aromatic natural products_Light.hdf5')
Polyketides_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Polyketides_Light.hdf5')
Polypyrroles_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Polypyrroles_Light.hdf5')
Carbohydrates_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Carbohydrates_Light.hdf5')
Terpenoids_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Terpenoids_Light.hdf5')
Polycyclic_aromatic_natural_products_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Polycyclic aromatic natural products_Light.hdf5')
#Terepenoid_subclass
Triterpenoids_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Terpenoids_Triterpenoids_Light.hdf5')
Sesquiterpenoids_model= keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Terpenoids_Sesquiterpenoids_Light.hdf5')
Diterpenoids_model= keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Terpenoids_Diterpenoids_Light.hdf5')
Monoterpenoids_model= keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Terpenoids_Monoterpenoids_Light.hdf5')
Sesterterpenoids_model= keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Terpenoids_Sesterterpenoids_Light.hdf5')
#Meroterpenoids_model = keras.models.load_model('Final_model/FP to Classifier/DNP_based_sub_classifier_Terpenoids_Meroterpenoids_Light.hdf5')

#Fingerprint generation
#(@ming if we want to use inchi as an input, inchi should be changed to SMILES and the SMILES should be standardized)

def FP(SMILES,radi):
    binary = np.zeros((2048*(radi)), int)
    formula = np.zeros((2048),int)
    misakinolide = Chem.MolFromSmiles(SMILES)
    misakinolide_H = Chem.AddHs(misakinolide)
    misa_bi_H = {}
    for r in range(radi+1):
        misa_fp_H = rdMolDescriptors.GetMorganFingerprintAsBitVect(misakinolide_H, radius=r, bitInfo=misa_bi_H, nBits = 2048)
        misa_bi_H_QC = []
        for i in misa_fp_H.GetOnBits():
            idx = misa_bi_H[i][0][0]
            radius_list = []
            for j in range(len(misa_bi_H[i])):
                atom_radi = misa_bi_H[i][j][1]
                radius_list.append(atom_radi) 
            atom = misakinolide_H.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            neigbor = [x.GetAtomicNum() for x in atom.GetNeighbors()]
            if r in radius_list: #and symbol == 'C' and 1 in neigbor:#radius = 2, atom = Carbon, H possessed Carbon
                misa_bi_H_QC.append(i)
        bits = misa_bi_H_QC
        for i in bits:
            if r == 0:
                formula[i] = len([k for k in misa_bi_H[i] if k[1]==r])
            else:
                binary[(2048*(r-1))+i] = 1
    
    
    
    return formula.reshape(1,2048),binary.reshape(1,4096)

def classifier(smiles):
    super_list = []
    sub_list = []
    terpene_sub_list = []
    try:
        
        fp = FP(smiles,2)
        n = np.where(super_model.predict(fp)[0]>=0.5)[0]
        
            
        for j in n:
            super_list.append([name for name, age in super_class.items() if age == j][0])

        for sup in super_list:
            if sup == 'Lignans':
                subs = np.where(Lignans_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Lignans'][m])
                #if sub_list == []:
                    #sub_list = classes['Lignans'][np.argmax(Lignans_model.predict(fp.reshape(1,6144)))]+'*'
                
            elif sup == 'Benzofuranoids':
                subs = np.where(Benzofuranoids_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Benzofuranoids'][m])
                #if sub_list == []:
                 #   sub_list = sub_list.append(classes['Benzofuranoids'][np.argmax(Benzofuranoids_model.predict(fp.reshape(1,6144)))]+'*'
                 
            elif sup == 'Steroids and Sterols':
                subs = np.where(Steroids_and_Sterols_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Steroids and Sterols'][m])
                #if sub_list == []:
                 #   sub_list = classes['Steroids and Sterols'][np.argmax(Steroids_and_Sterols_model.predict(fp.reshape(1,6144)))]+'*'
                
            elif sup == 'Aminoacids and peptides':
                subs = np.where(Aminoacids_and_peptides_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Aminoacids and peptides'][m])
                #if sub_list == []:
                 #   sub_list = classes['Aminoacids and peptides'][np.argmax(Aminoacids_and_peptides_model.predict(fp.reshape(1,6144)))]+'*'
                
            elif sup == 'Flavonoids':
                subs = np.where(Flavonoids_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Flavonoids'][m])
                #if sub_list == []:
                 #   sub_list = classes['Flavonoids'][np.argmax(Flavonoids_model.predict(fp.reshape(1,6144)))]+'*'
                 
            elif sup == 'Aliphatic natural products':
                subs = np.where(Aliphatic_natural_products_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Aliphatic natural products'][m])
                #if sub_list == []:
                    #sub_list = classes['Aliphatic natural products'][np.argmax(Lipids_model.predict(fp.reshape(1,6144)))]+'*'
                    
            elif sup == 'Oxygenheterocycles':
                subs = np.where(Oxygenheterocycles_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Oxygenheterocycles'][m])
                #if sub_list == []:
                 #   sub_list = classes['Lipids'][np.argmax(Lipids_model.predict(fp.reshape(1,6144)))]+'*'
                    
            elif sup == 'Benzopyranoids':
                subs = np.where(Benzopyranoids_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Benzopyranoids'][m])
                #if sub_list == []:
                 #   sub_list = classes['Benzopyranoids'][np.argmax(Benzopyranoids_model.predict(fp.reshape(1,6144)))]+'*'
                    
            elif sup == 'Alkaloids':
                subs = np.where(Alkaloids_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Alkaloids'][m])
                #if sub_list == []:
                 #   sub_list = classes['Alkaloids'][np.argmax(Alkaloids_model.predict(fp.reshape(1,6144)))]+'*'
             
            elif sup == 'Tannins':
                subs = np.where(Tannins_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Tannins'][m])
             #   if sub_list == []:
              #      sub_list = classes['Tannins'][np.argmax(Tannins_model.predict(fp.reshape(1,6144)))]+'*'
                
            elif sup == 'Simple aromatic natural products':
                subs = np.where(Simple_aromatic_natural_products_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Simple aromatic natural products'][m])
               # if sub_list == []:
                #    sub_list = classes['Simple aromatic natural products'][np.argmax(Simple_aromatic_natural_products_model.predict(fp.reshape(1,6144)))]+'*'
                 
            elif sup == 'Polyketides':
                subs = np.where(Polyketides_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Polyketides'][m])
                #if sub_list == []:
                 #   sub_list = classes['Polyketides'][np.argmax(Polyketides_model.predict(fp.reshape(1,6144)))]+'*'
                    
            elif sup == 'Polypyrroles':
                subs = np.where(Polypyrroles_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Polypyrroles'][m])
                #if sub_list == []:
                 #   sub_list = classes['Polypyrroles'][np.argmax(Polypyrroles_model.predict(fp.reshape(1,6144)))]+'*'
            
            elif sup == 'Carbohydrates':
                subs = np.where(Carbohydrates_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Carbohydrates'][m])
                    
            
            elif sup == 'Terpenoids':
                subs = np.where(Terpenoids_model.predict(fp)[0]>=0.5)[0]
                for m in subs:
                    sub_list.append(classes['Terpenoids'][m])
               # if sub_list == []:
                #    sub_list = classes['Terpenoids'][np.argmax(Terpenoids_model.predict(fp.reshape(1,6144)))]+'*'
                
                for n in sub_list:
                    if n == 'Triterpenoids':
                        subs_terpene = np.where(Triterpenoids_model.predict(fp)[0]>=0.5)[0]
                        for o in subs_terpene:
                            terpene_sub_list.append(classes_terpene['Triterpenoids'][o])
                 #       if terpene_sub_list == []:
                  #          terpene_sub_list = classes_terpene['Triterpenoids'][np.argmax(Triterpenoids_model.predict(fp.reshape(1,6144)))]+'*'
                    elif n == 'Diterpenoids':
                        subs_terpene = np.where(Diterpenoids_model.predict(fp)[0]>=0.5)[0]
                        for o in subs_terpene:
                            terpene_sub_list.append(classes_terpene['Diterpenoids'][o])
                        #if terpene_sub_list == []:
                         #   terpene_sub_list = classes_terpene['Diterpenoids'][np.argmax(Diterpenoids_model.predict(fp.reshape(1,6144)))]+'*'
                    elif n == 'Sesquiterpenoids':
                        subs_terpene = np.where(Sesquiterpenoids_model.predict(fp)[0]>=0.5)[0]
                        for o in subs_terpene:
                            terpene_sub_list.append(classes_terpene['Sesquiterpenoids'][o])
                        #if terpene_sub_list == []:
                         #   terpene_sub_list = classes_terpene['Sesquiterpenoids'][np.argmax(Sesquiterpenoids_model.predict(fp.reshape(1,6144)))]+'*'
                    elif n == 'Sesterterpenoids':
                        subs_terpene = np.where(Sesterterpenoids_model.predict(fp)[0]>=0.5)[0]
                        for o in subs_terpene:
                            terpene_sub_list.append(classes_terpene['Sesterterpenoids'][o])
                        #if terpene_sub_list == []:
                         #   terpene_sub_list = classes_terpene['Sesterterpenoids'][np.argmax(Sesterterpenoids_model.predict(fp.reshape(1,6144)))]+'*'
                    elif n == 'Monoterpenoids':
                        subs_terpene = np.where(Monoterpenoids_model.predict(fp)[0]>=0.5)[0]
                        for o in subs_terpene:
                            terpene_sub_list.append(classes_terpene['Monoterpenoids'][o])
                        #if terpene_sub_list == []:
                         #   terpene_sub_list = classes_terpene['Monoterpenoids'][np.argmax(Monoterpenoids_model.predict(fp.reshape(1,6144)))]+'*'
                    #elif n == 'Meroterpenoids':
                        #subs_terpene = np.where(Meroterpenoids_model.predict(fp)[0]>=0.5)[0]
                        #for o in subs_terpene:
                            #terpene_sub_list.append(classes_terpene['Meroterpenoids'][o])
                        #if terpene_sub_list == []:
                         #   terpene_sub_list = classes_terpene['Monoterpenoids'][np.argmax(Monoterpenoids_model.predict(fp.reshape(1,6144)))]+'*'
                            
            elif sup == 'Polycyclic aromatic natural products':
                subs = np.where(Polycyclic_aromatic_natural_products_model.predict(fp)[0]>0.5)[0]
                for m in subs:
                    sub_list.append(classes['Polycyclic aromatic natural products'][m])
               # if sub_list == []:
                #    sub_list = classes['Polycyclic aromatic natural products'][np.argmax(Polycyclic_aromatic_natural_products_model.predict(fp.reshape(1,6144)))]+'*'
                
    except:
        print('error')
        pass
    
    return super_list,sub_list, terpene_sub_list
