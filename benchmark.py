# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:17:14 2017

@author: JTay
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = './BASE/'
np.random.seed(0)

cancer = pd.read_hdf('./output/BASE/datasets.hdf','cancer')
cancerX = cancer.drop('class',1).copy().values
cancerY = cancer['class'].copy().values

contra = pd.read_hdf('./output/BASE/datasets.hdf','contra')        
contraX = contra.drop('class',1).copy().values
contraY = contra['class'].copy().values

contraX = StandardScaler().fit_transform(contraX)
cancerX = StandardScaler().fit_transform(cancerX)

#%% benchmarking for chart type 2

grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5, return_train_score = True)

gs.fit(contraX, contraY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'contra NN bmk.csv')


mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5, return_train_score = True)

gs.fit(cancerX, cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer NN bmk.csv')
#raise