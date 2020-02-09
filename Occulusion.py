import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import math
import pylab
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from math import sqrt
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib as mpl # 그래프 그리는 모듈
import matplotlib.pyplot as plt # 그래프를 그리는 모듈
import os
from tqdm import tqdm
import cairosvg
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

def Occlusion_exp(model, image, occluding_size, occluding_pixel, occluding_stride, name, SMILES):
    
    qc = np.zeros((128,128),int)
    try:
        qc = (image[0,:,:,0] + image[0,:,:,1]>0).astype(int)
    except:
        qc = (image[0,:,:,0])
    
    out_total = model.predict(image)
    out_total = out_total[0][0]
    # Getting the index of the winning class:
    score = cosine_mat(HSQC_bit_bi(SMILES,2,2048),out_total)
    print('Matching Score:',score)
    try:
        os.makedirs(os.path.join('Attention/{}_{}'.format(name,round(score,2))))
    except:
        None
    
    index_object_list = [i for i, j in enumerate(out_total) if j >= 0.5]
    height, width, _ = image.shape[1:4]
    output_height = int(math.ceil((height-occluding_size) / occluding_stride + 1))
    output_width = int(math.ceil((width-occluding_size) / occluding_stride + 1))
    heatmap = np.zeros((output_height, output_width, 2048))
        
    for h in range(output_height):
        for w in range(output_width):
            # Occluder region:
            h_start = h * occluding_stride
            w_start = w * occluding_stride
            h_end = min(height, h_start + occluding_size)
            w_end = min(width, w_start + occluding_size)
            # Getting the image copy, applying the occluding window and classifying it again:
            input_image = copy.copy(image)
            
            if np.sum(input_image[0,h_start:h_end, w_start:w_end,:]) != 0:
                input_image[0,h_start:h_end, w_start:w_end,:] =  occluding_pixel            
                out = model.predict(input_image)
                out = out[0][0]
                for index_object in index_object_list:
                # It's possible to evaluate the VGG-16 sensitivity to a specific object.
                # To do so, you have to change the variable "index_object" by the index of
                # the class of interest. The VGG-16 output indices can be found here:
                # https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt
                    prob = (out[index_object]) 
                    heatmap[h,w, index_object] = prob
            elif np.sum(input_image[0,h_start:h_end, w_start:w_end,:]) == 0:
                for index_object in index_object_list:
                    heatmap[h,w,index_object] = out_total[index_object]
        
       
    SMILES = SMILES
    r = 2
    bits = 2048
    misakinolide = Chem.MolFromSmiles(SMILES)
    misakinolide_H = Chem.AddHs(misakinolide)
    misa_bi_H = {}
    misa_fp_H = rdMolDescriptors.GetMorganFingerprintAsBitVect(misakinolide_H, radius=r, bitInfo=misa_bi_H, nBits = bits)
    misa_bi_H_QC = []
    misa_bi_H_QC_r = []
    for i in misa_fp_H.GetOnBits():
        idx = misa_bi_H[i][0][0]
        radius = misa_bi_H[i][0][1]
        atom = misakinolide_H.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        neigbor = [x.GetAtomicNum() for x in atom.GetNeighbors()]
        misa_bi_H_QC.append(i)
        if radius >= 1 and symbol != 'H':#radius = 2, atom = Carbon, H possessed Carbon
            misa_bi_H_QC_r.append(i)
    misa_tpls_H = [(misakinolide_H,x,misa_bi_H) for x in misa_bi_H_QC]       
    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(misakinolide, kekulize=True)    
    
    
    for bit in index_object_list:
        
        heatmap_bit = cv2.resize(heatmap[:,:,bit], (128, 128))
        C_scale, H_scale = heatmap_bit.shape[0], heatmap_bit.shape[1]
        plt.figure()
        ax = plt.axes()

        plt.imshow(qc, cmap=mpl.colors.ListedColormap([(0.2, 0.4, 0.6, 0),'black']))
        plt.imshow(heatmap_bit, cmap='jet', alpha = 0.7)
        plt.colorbar(extend='both')
        ax.set_ylim(C_scale,0)
        ax.set_xlim(0,H_scale)
        plt.axis()
        plt.xticks(np.arange(H_scale,step=H_scale/12), (list(range(12,-1,-1))))
        plt.yticks(np.arange(C_scale,step=C_scale/12), (list(range(0,260,20))))
        ax.set_xlabel('1H [ppm]')
        ax.set_ylabel('13C [ppm]')
        plt.grid(True,linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.text(4, 10, '{}'.format(bit), ha='left', wrap=True)
        plt.savefig('Attention/{}_{}/bit_{}.png'.format(name,round(score,2),bit), dpi=600)
        plt.close()       

    # getting highlight of atom

        
        bit = bit
        try:
            highlight=[]
            radius = []

            for i in range(len(misa_tpls_H[0][2][bit])):
                highlight.append(misa_tpls_H[0][2][bit][i][0])
                r = misa_tpls_H[0][2][bit][i][1]
                radius.append(r+1)
            highlight_r = dict(zip(highlight, radius))
            drawer = rdMolDraw2D.MolDraw2DSVG(400,200)
            drawer.DrawMolecule(mol,highlightAtoms=highlight,highlightBonds=[],highlightAtomRadii=highlight_r )
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText().replace('svg:','')
            cairosvg.svg2png(svg, 
                             write_to='Attention/{}_{}/sbit_{}.png'.format(name,round(score,2),bit), 
                             dpi = 600, output_width=2000, output_height=1000
                            )
            
        except:
            None
    
    misa_bi_H_QC_r = set(misa_bi_H_QC_r)
    index_object_list = set(index_object_list)
    loc = misa_bi_H_QC_r&index_object_list
    highlight=[]
    for bit in loc:
        for i in range(len(misa_tpls_H[0][2][bit])):
            highlight.append(misa_tpls_H[0][2][bit][i][0])
    highlight = set(highlight)    
    drawer = rdMolDraw2D.MolDraw2DSVG(400,200)
    drawer.DrawMolecule(mol,highlightAtoms=highlight,highlightAtomRadii=[] )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    cairosvg.svg2png(svg, write_to='Attention/{}_{}/{}_map.png'.format(name,round(score,2),name), dpi = 600, output_width=2000, output_height=1000)
            
