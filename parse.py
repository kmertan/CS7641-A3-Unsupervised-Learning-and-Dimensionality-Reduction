# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:39:27 2017

@author: jtay
"""

import pandas as pd
import numpy as np
import os 
import sklearn.model_selection as ms

for d in ['BASE','RP','PCA','ICA','RF']:
    n = './output/{}/'.format(d)
    if not os.path.exists(n):
        os.makedirs(n)

OUT = './output/BASE/'

contra = pd.read_csv('./data/cmc.data.txt')      
col_names_contra = ['wifes_age', 'wifes_edu', 'husbs_edu', 'num_children_born', 'wifes_religion', 'wife_employed', 'husbs_occup', 'SOL_index', 'media_expose', 'class']
contra.columns = col_names_contra
contra[col_names_contra] = contra[col_names_contra].astype(np.int64)
contra = pd.get_dummies(contra, columns = ['husbs_occup'], prefix = 'husbs_occup')

contraX = contra.drop('class',1).copy().values
contraY = contra['class'].copy().values
#print(contraX)

cancer = pd.read_csv('./data/breast_cancer.csv')  
cancer = pd.get_dummies(cancer, columns = ['class'], prefix = 'class')
cancer['class'] = cancer['class_2.0']
cancer.drop(['class_2.0', 'class_4.0'], axis = 1, inplace = True)
cancer = cancer.astype(np.int64)

#contra.to_hdf(OUT+'datasets.hdf','contra',complib='blosc',complevel=9)
#cancer.to_hdf(OUT+'datasets.hdf','cancer',complib='blosc',complevel=9)

cancerX = cancer.drop('class',1).copy().values
cancerY = cancer['class'].copy().values
#
contra_trgX, contra_tstX, contra_trgY, contra_tstY = ms.train_test_split(contraX, contraY, test_size=0.3, random_state=0,stratify=contraY)     
#
cancer_trgX, cancer_tstX, cancer_trgY, cancer_tstY = ms.train_test_split(cancerX, cancerY, test_size=0.3, random_state=0,stratify=cancerY)     
#
contraX = pd.DataFrame(contra_trgX)
contraY = pd.DataFrame(contra_trgY)
contraY.columns = ['class']

contraX2 = pd.DataFrame(contra_tstX)
contraY2 = pd.DataFrame(contra_tstY)
contraY2.columns = ['class']
#
contra1 = pd.concat([contraX,contraY],1)
contra1 = contra1.dropna(axis=1,how='all')
contra1.to_hdf(OUT+'datasets.hdf','contra',complib='blosc',complevel=9)
#
contra2 = pd.concat([contraX2,contraY2],1)
contra2 = contra2.dropna(axis=1,how='all')
contra2.to_hdf(OUT+'datasets.hdf','contra_test',complib='blosc',complevel=9)
#
#
#
cancerX = pd.DataFrame(cancer_trgX)
cancerY = pd.DataFrame(cancer_trgY)
cancerY.columns = ['class']
#
cancerX2 = pd.DataFrame(cancer_tstX)
cancerY2 = pd.DataFrame(cancer_tstY)
cancerY2.columns = ['class']
#
cancer1 = pd.concat([cancerX,cancerY],1)
cancer1 = cancer1.dropna(axis=1,how='all')
cancer1.to_hdf(OUT+'datasets.hdf','cancer',complib='blosc',complevel=9)
#
cancer2 = pd.concat([cancerX2,cancerY2],1)
cancer2 = cancer2.dropna(axis=1,how='all')
cancer2.to_hdf(OUT+'datasets.hdf','cancer_test',complib='blosc',complevel=9)