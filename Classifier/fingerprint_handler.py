import pandas as pd
from tqdm import tqdm
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

#Fingerprint generation
#(@ming if we want to use inchi as an input, inchi should be changed to SMILES and the SMILES should be standardized)
def calculate_fingerprint(smiles, radi):
    binary = np.zeros((2048*(radi)), int)
    formula = np.zeros((2048), int)

    mol = Chem.MolFromSmiles(smiles)
    #mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = Chem.AddHs(mol)
    #mol = Chem.AddHs(mol, explicitOnly=True, addCoords=True, onlyOnAtoms=True)
    
    mol_bi = {}
    for r in range(radi+1):
        mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=r, bitInfo=mol_bi, nBits = 2048)
        mol_bi_QC = []
        for i in mol_fp.GetOnBits():
            idx = mol_bi[i][0][0]
            radius_list = []
            num_ = len(mol_bi[i])
            for j in range(num_):
                if mol_bi[i][j][1] == r:
                    mol_bi_QC.append(i)
                    break

        if r == 0:
            for i in mol_bi_QC:
                formula[i] = len([k for k in mol_bi[i] if k[1]==0])
        else:
            for i in mol_bi_QC:
                binary[(2048*(r-1))+i] = 1
    formula[1652] = 0 #removing the number of protons from fingerprint
    
    return formula.reshape(1,2048), binary.reshape(1,4096)

