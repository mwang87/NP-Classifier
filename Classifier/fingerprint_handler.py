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
    formula = np.zeros((2048),int)
    mol = Chem.MolFromSmiles(smiles)
    
    mol = Chem.AddHs(mol)
    mol_bi = {}
    for r in range(radi+1):
        mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=r, bitInfo=mol_bi, nBits = 2048)
        mol_bi_QC = []
        for i in mol_fp.GetOnBits():
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
                binary[(2048*(r-1))+i] = len([k for k in mol_bi[i] if k[1]==r])
    
    
    
    return formula.reshape(1,2048),binary.reshape(1,4096)


def _isglycoside(smiles): #now it is expressed as boolean but can be changed to any format
    #sugar1 = Chem.MolFromSmarts('[OX2;$([r5]1@C@C@C(O)@C1),$([r6]1@C@C@C(O)@C(O)@C1)]')
    sugar2 = Chem.MolFromSmarts('[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]')
    sugar3 = Chem.MolFromSmarts('[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C(O)@C1)]')
    sugar4 = Chem.MolFromSmarts('[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),$([r6]1@C(!@[OX2H1])@C@C@C@C1)]')
    #sugar5 = Chem.MolFromSmarts('[OX2;$([r5]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]')
    #sugar6 = Chem.MolFromSmarts('[OX2;$([r5]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]')
    mol = Chem.MolFromSmiles(smiles)
    try:
        if (mol.HasSubstructMatch(sugar2) or
            mol.HasSubstructMatch(sugar3) or
            mol.HasSubstructMatch(sugar4)) :
            return True 
        else:
            return False 
    except:
        return 'Input_error'
