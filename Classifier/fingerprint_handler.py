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
def calculate_fingerprint(SMILES, radi):
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