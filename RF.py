

#%% Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg,ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    out = './output/RF/'
    
    np.random.seed(0)
    cancer = pd.read_hdf('./output/BASE/datasets.hdf','cancer')
    cancerX = cancer.drop('class',1).copy().values
    cancerY = cancer['class'].copy().values
    
    contra = pd.read_hdf('./output/BASE/datasets.hdf','contra')        
    contraX = contra.drop('class',1).copy().values
    contraY = contra['class'].copy().values
    
    contraX = StandardScaler().fit_transform(contraX)
    cancerX = StandardScaler().fit_transform(cancerX)
    
    clusters =  range(2, 10)
    
    dims_contra = range(1, 12)
    dims_cancer = range(1, 10)
    
    #%% data for 1
    
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
    fs_contra = rfc.fit(contraX,contraY).feature_importances_ 
    fs_cancer = rfc.fit(cancerX,cancerY).feature_importances_ 
    
    tmp = pd.Series(np.sort(fs_contra)[::-1])
    tmp.to_csv(out+'contra scree.csv')
    
    indices = np.argsort(fs_contra)[::-1]
    for f in range(contraX.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], fs_contra[indices[f]]))
    
    tmp = pd.Series(np.sort(fs_cancer)[::-1])
    tmp.to_csv(out+'cancer scree.csv')
    
    indices = np.argsort(fs_cancer)[::-1]
    for f in range(cancerX.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], fs_cancer[indices[f]]))
    
    #%% Data for 2
    filtr = ImportanceSelect(rfc)
    grid ={'filter__n':dims_contra,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(contraX,contraY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'contra dim red.csv')
    
    
    grid ={'filter__n':dims_cancer,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}  
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(cancerX,cancerY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'cancer dim red.csv')
#    raise
    #%% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 7
    filtr = ImportanceSelect(rfc,dim)
    
    contraX2 = filtr.fit_transform(contraX,contraY)
    contra2 = pd.DataFrame(np.hstack((contraX2,np.atleast_2d(contraY).T)))
    cols = list(range(contra2.shape[1]))
    cols[-1] = 'class'
    contra2.columns = cols
    contra2.to_hdf(out+'datasets.hdf','contra',complib='blosc',complevel=9)
    
    dim = 6
    filtr = ImportanceSelect(rfc,dim)
    cancerX2 = filtr.fit_transform(cancerX,cancerY)
    cancer2 = pd.DataFrame(np.hstack((cancerX2,np.atleast_2d(cancerY).T)))
    cols = list(range(cancer2.shape[1]))
    cols[-1] = 'class'
    cancer2.columns = cols
    cancer2.to_hdf(out+'datasets.hdf','cancer',complib='blosc',complevel=9)