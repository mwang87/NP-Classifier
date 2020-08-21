#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Dataaugmentation
import copy
#rdkit
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

#List of hydroxyl groups in the molecule
def list_of_hydroxy(mol):

    hydrogens = []
    atom_num = mol.GetNumAtoms()
    for idx in range(atom_num):
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        neigbor = [x.GetAtomicNum() for x in atom.GetNeighbors()]
        if symbol == 'H' and 8 in neigbor:
            hydrogens.append(atom)
    return hydrogens

#List of methoxyl group in the molecule
def list_of_methoxy(mol):

    methoxys = []
    atom_num = mol.GetNumAtoms()
    for idx in range(atom_num):
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        neigbor = [x.GetAtomicNum() for x in atom.GetNeighbors()]
        if symbol == 'C' and neigbor == [8]:
            methoxys.append(atom)
    return methoxys

#Generate methoxylated analogues
def methoxy(smiles):
    smiles_list = [smiles]
    mol = Chem.MolFromSmiles(smiles) 
    list_homologues = []
    mol = Chem.AddHs(mol)
    list_len = len(list_of_hydroxy(mol))
    for j in range(list_len):
        
        mol_new = copy.copy(mol)
        list_of_hydroxy(mol_new)[j].SetAtomicNum(6)
        
        mol_new = Chem.RemoveHs(mol_new)
        
        list_homologues.append(mol_new)
    smiles_list2 = [Chem.MolToSmiles(mol_homo) for mol_homo in list_homologues]
    smiles_list = smiles_list+smiles_list2
    return smiles_list

#Generate demethoxylated analogues
def demethoxy(smiles):
    smiles_list = [smiles]
    mol = Chem.MolFromSmiles(smiles) 
    list_homologues = []
    list_len = len(list_of_methoxy(mol))
    for j in range(list_len):
        
        mol_new = copy.copy(mol)
        list_of_methoxy(mol_new)[j].SetAtomicNum(1)
        
        mol_new = Chem.RemoveHs(mol_new)
        
        list_homologues.append(mol_new)
    smiles_list2 = [Chem.MolToSmiles(mol_homo) for mol_homo in list_homologues]
    smiles_list = smiles_list+smiles_list2
    return smiles_list

