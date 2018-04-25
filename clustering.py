# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:38:28 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys

out = './output/{}/'.format(sys.argv[1])
#out = './output/clustering/'

np.random.seed(0)

cancer = pd.read_hdf(out + 'datasets.hdf','cancer')
cancerX = cancer.drop('class',1).copy().values
cancerY = cancer['class'].copy().values

contra = pd.read_hdf(out + 'datasets.hdf','contra')        
contraX = contra.drop('class',1).copy().values
contraY = contra['class'].copy().values

contraX = StandardScaler().fit_transform(contraX)
cancerX = StandardScaler().fit_transform(cancerX)

clusters = range(1, 10)

#%% Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(contraX)
    gmm.fit(contraX)
    SSE[k]['contra'] = km.score(contraX)
    ll[k]['contra'] = gmm.score(contraX)    
    acc[k]['contra']['Kmeans'] = cluster_acc(contraY,km.predict(contraX))
    acc[k]['contra']['GMM'] = cluster_acc(contraY,gmm.predict(contraX))
    adjMI[k]['contra']['Kmeans'] = ami(contraY,km.predict(contraX))
    adjMI[k]['contra']['GMM'] = ami(contraY,gmm.predict(contraX))
    
    km.fit(cancerX)
    gmm.fit(cancerX)
    SSE[k]['cancer'] = km.score(cancerX)
    ll[k]['cancer'] = gmm.score(cancerX)
    acc[k]['cancer']['Kmeans'] = cluster_acc(cancerY,km.predict(cancerX))
    acc[k]['cancer']['GMM'] = cluster_acc(cancerY,gmm.predict(cancerX))
    adjMI[k]['cancer']['Kmeans'] = ami(cancerY,km.predict(cancerX))
    adjMI[k]['cancer']['GMM'] = ami(cancerY,gmm.predict(cancerX))
    print(k, clock()-st)

    
## Keith Mertan: Adding cluster outputs for best parameters and saving at the end of the file

## Cancer data first

km = kmeans(random_state=5)
gmm = GMM(random_state=5)

if sys.argv[1] == 'BASE':
    km.set_params(n_clusters = 5)
    gmm.set_params(n_components = 5)
    
if sys.argv[1] == 'PCA':
    km.set_params(n_clusters = 8)
    gmm.set_params(n_components = 8)
    
if sys.argv[1] == 'ICA':
    km.set_params(n_clusters = 4)
    gmm.set_params(n_components = 3)
    
if sys.argv[1] == 'RP':
    km.set_params(n_clusters = 4)
    gmm.set_params(n_components = 6)

if sys.argv[1] == 'RF':
    km.set_params(n_clusters = 4)
    gmm.set_params(n_components = 7)
    
km.fit(cancerX)
gmm.fit(cancerX)

cancer_gmm_clusters = km.predict(cancerX)
cancer_km_clusters = gmm.predict(cancerX)

cancer_gmm_clusters = np.atleast_2d(cancer_gmm_clusters).T
cancer_km_clusters = np.atleast_2d(cancer_km_clusters).T

## Contraceptive data second

km = kmeans(random_state=5)
gmm = GMM(random_state=5)

if sys.argv[1] == 'BASE':
    km.set_params(n_clusters = 7)
    gmm.set_params(n_components = 7)
    
if sys.argv[1] == 'PCA':
    km.set_params(n_clusters = 8)
    gmm.set_params(n_components = 8)
    
if sys.argv[1] == 'ICA':
    km.set_params(n_clusters = 7)
    gmm.set_params(n_components = 9)
    
if sys.argv[1] == 'RP':
    km.set_params(n_clusters = 6)
    gmm.set_params(n_components = 9)

if sys.argv[1] == 'RF':
    km.set_params(n_clusters = 5)
    gmm.set_params(n_components = 7)
    
km.fit(contraX)
gmm.fit(contraX)

contra_gmm_clusters = km.predict(contraX)
contra_km_clusters = gmm.predict(contraX)

contra_gmm_clusters = np.atleast_2d(contra_gmm_clusters).T
contra_km_clusters = np.atleast_2d(contra_km_clusters).T
    
##

SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)


SSE.to_csv(out+'SSE.csv')
ll.to_csv(out+'logliklihood.csv')
acc.ix[:,:,'cancer'].to_csv(out+'cancer acc.csv')
acc.ix[:,:,'contra'].to_csv(out+'contra acc.csv')
adjMI.ix[:,:,'cancer'].to_csv(out+'cancer adjMI.csv')
adjMI.ix[:,:,'contra'].to_csv(out+'contra adjMI.csv')


#%% NN fit data (2,3)

grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10)

gs.fit(contraX,contraY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'contra cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(contraX,contraY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'contra cluster GMM.csv')

## Keith Mertan: Adding gridsearches for cluster labels alone (above are kmeans distances and gmm predicted probabilities)
## and for cluster labels in addition to the given data

grid ={'km__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('km',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(contra_km_clusters,contraY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'contra cluster KM CLUSTER ONLY.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(contra_gmm_clusters,contraY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'contra cluster GMM CLUSTER ONLY.csv')


grid ={'km__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('km',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(np.hstack((contraX, contra_km_clusters)),contraY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'contra cluster KM CLUSTERS AND DATA.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(np.hstack((contraX, contra_gmm_clusters)),contraY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'contra cluster GMM CLUSTERS WITH DATA.csv')

##




grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(cancerX,cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(cancerX,cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer cluster GMM.csv')

## Keith Mertan: Adding gridsearches for cluster labels alone (above are kmeans distances and gmm predicted probabilities)
## and for cluster labels in addition to the given data

grid ={'km__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('km',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(cancer_km_clusters,cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer cluster KM CLUSTER ONLY.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(cancer_gmm_clusters,cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer cluster GMM CLUSTER ONLY.csv')

grid ={'km__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('km',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(np.hstack((cancerX, cancer_km_clusters)),cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer cluster KM CLUSTER AND DATA.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(np.hstack((cancerX, cancer_gmm_clusters)),cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer cluster GMM CLUSTER AND DATA.csv')

##


# %% For chart 4/5
contraX2D = TSNE(verbose=10,random_state=5).fit_transform(contraX)
cancerX2D = TSNE(verbose=10,random_state=5).fit_transform(cancerX)

contra2D = pd.DataFrame(np.hstack((contraX2D,np.atleast_2d(contraY).T,contra_gmm_clusters,contra_km_clusters)),columns=['x','y','target','gmm_cluster', 'km_cluster'])
cancer2D = pd.DataFrame(np.hstack((cancerX2D,np.atleast_2d(cancerY).T,cancer_gmm_clusters,cancer_km_clusters)),columns=['x','y','target','gmm_cluster', 'km_cluster'])

contra2D.to_csv(out+'contra2D.csv')
cancer2D.to_csv(out+'cancer2D.csv')