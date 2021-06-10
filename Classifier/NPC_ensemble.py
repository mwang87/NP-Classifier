from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from tqdm import tqdm
import numpy as np
import itertools 

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

def isglycoside(smiles): #now it is expressed as boolean but can be changed to any format
    hexa_pyranose = Chem.MolFromSmarts('[O]C1C([O])C([O])C(C[O])OC1[*]')
    penta_furanose = Chem.MolFromSmarts('[O]CC1OC([*])C([O])C1[O]')
    mol = Chem.MolFromSmiles(smiles)
    try:
        if mol.HasSubstructMatch(hexa_pyranose) or mol.HasSubstructMatch(penta_furanose):
            return True 
        else:
            return False 
    except:
        return 'Input_error'


def chain(*iterables):
    # chain('ABC', 'DEF') --> ['A', 'B', 'C', 'D', 'E', 'F']
    for it in iterables:
        for element in it:
            yield element
            
def from_iterable(iterables):
    # chain.from_iterable(['ABC', 'DEF']) --> ['A', 'B', 'C', 'D', 'E', 'F']
    for it in iterables:
        for element in it:
            yield element
    
import json
#Loading dictionary of super_class, class and subclass(terpene)

def main():
    with open('dict/index_v1.json','r') as total:
        index = json.load(total)

    index_class = list(index['Class'].keys())
    index_superclass = list(index['Superclass'].keys())
    index_pathway = list(index['Pathway'].keys())

    # This part may take memory leakage
    #Super_class
    model_class = keras.models.load_model('model/NP_classifier_class_V25.hdf5')
    model_super = keras.models.load_model('model/NP_classifier_superclass_V25.hdf5')
    model_pathway = keras.models.load_model('model/NP_classifier_pathway_V25.hdf5')

def classifier(smiles):
    
    class_result=[]
    superclass_result = []
    pathway_result = []

    path_from_class  = []
    path_from_superclass = []

    glycoside = isglycoside(smiles)
    
    fp = FP(smiles,2)
    pred_class = model_class.predict(fp)[0]
    pred_super = model_super.predict(fp)[0]
    pred_path = model_pathway.predict(fp)[0]        

    n_class = list(np.where(pred_class>=0.5)[0])
    n_super = list(np.where(pred_super>=0.5)[0])
    n_path = list(np.where(pred_path>=0.5)[0])

    if n_path == []:
        n_path = [np.argmax(pred_path)]

    for i in n_class:
        path_from_class = path_from_class+index['Class_hierarchy'][str(i)]['Pathway']
    for j in n_super:
        path_from_superclass = path_from_superclass+index['Super_hierarchy'][str(j)]['Pathway']

    path_from_class = list(set(path_from_class))
    path_from_superclass = list(set(path_from_superclass))

    path_for_vote = n_path+path_from_class+path_from_superclass
    path = list(set([ k for k in path_for_vote if path_for_vote.count(k) ==3]))

    if path == []:
        path = list(set([ k for k in path_for_vote if path_for_vote.count(k) ==2]))
        if len(path)>1:
                path_for_vote = n_path+path_from_superclass
                n_class = []
                path = list(set([ k for k in path_for_vote if path_for_vote.count(k) ==2])) 
    if path == []:
        for w in n_path:
            pathway_result.append(index_pathway[w])
            return pathway_result,superclass_result,class_result,glycoside


    else: #path != []
        if set(n_path) & set(path) != set():
            if set(path) & set(path_from_superclass) != set():
                n_super = [ l for l in n_super if set(path)& set(index['Super_hierarchy'][str(l)]['Pathway']) != set()]
                if n_super == []:
                    n_class = [ m for m in n_class if set(path) & set(index['Class_hierarchy'][str(m)]['Pathway']) != set() ]
                    n_super = [index['Class_hierarchy'][str(n)]['Superclass'] for n in n_class]
                    n_super = list(set(itertools.chain.from_iterable(n_super)))

                elif len(n_super) > 1: #super != []
                    n_class = [ u for u in n_class if set(path) & set(index['Class_hierarchy'][str(u)]['Pathway']) != set() ]
                    if n_class != []:
                        n_super = [index['Class_hierarchy'][str(v)]['Superclass'] for v in n_class]
                        n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
                        n_path = list(set(itertools.chain.from_iterable(n_path)))
                        n_super = list(set(itertools.chain.from_iterable(n_super)))

                    elif len(path)==1:
                        n_super = [np.argmax(pred_super)]
                        n_class = [ m for m in [np.argmax(pred_class)] if set(n_super) & set(index['Class_hierarchy'][str(m)]['Superclass']) != set() ]


                else:
                    n_class = [ o for o in n_class if set(n_super) & set(index['Class_hierarchy'][str(o)]['Superclass'])!=set() ]
                    if n_class == []:
                        n_class = [ m for m in [np.argmax(pred_class)] if set(n_super) & set(index['Class_hierarchy'][str(m)]['Superclass']) != set() ]
            else:
                n_class = [ p for p in n_class if  set(path) & set(index['Class_hierarchy'][str(p)]['Pathway']) !=set() ]
                n_super = [index['Class_hierarchy'][str(q)]['Superclass'] for q in n_class]

                n_super = list(set(itertools.chain.from_iterable(n_super)))

        else:
            n_super = [ l for l in n_super if set(path) & set(index['Super_hierarchy'][str(l)]['Pathway']) != set()]
            if n_super == []:
                n_class = [ m for m in n_class if set(path) & set(index['Class_hierarchy'][str(m)]['Pathway']) != set()]
                n_super = [index['Class_hierarchy'][str(n)]['Superclass'] for n in n_class]
                n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
                n_path = list(set(itertools.chain.from_iterable(n_path)))
                n_super = list(set(itertools.chain.from_iterable(n_super)))


            elif len(n_super) > 1: #super != []
                n_class = [ u for u in n_class if set(path) & set(index['Class_hierarchy'][str(u)]['Pathway']) != set()]
                n_super = [index['Class_hierarchy'][str(v)]['Superclass'] for v in n_class]
                n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
                n_path = list(set(itertools.chain.from_iterable(n_path)))
                n_super = list(set(itertools.chain.from_iterable(n_super)))


            else:
                n_class = [ o for o in n_class if set(path) & set(index['Class_hierarchy'][str(o)]['Pathway']) != set() ]
                n_super = [index['Class_hierarchy'][str(v)]['Superclass'] for v in n_class]
                n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
                n_path = list(set(itertools.chain.from_iterable(n_path)))
                n_super = list(set(itertools.chain.from_iterable(n_super)))

    for r in path:
        pathway_result.append(index_pathway[r])
    for s in n_super:
        superclass_result.append(index_superclass[s])
    for t in n_class:
        class_result.append(index_class[t])
    
    
    return pathway_result,superclass_result,class_result,glycoside #three class results and glycoside checker result (True/False)
    


