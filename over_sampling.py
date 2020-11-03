# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:04:31 2020

@author: lwzjc
"""
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle

def MLSMOTE(x, y, k_neighbors=5, criterion_rate=1.5, maxstep=10):
    """
    Synthetic Minority Over-sampling Technique on multi-label dataset
    resample only the samples with minority classes

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features).
    y : ndarray, shape (n_samples, n_labels). 
        y(i,j)=0 if i-th sample don't have j-th label, else 1
    
    k_neighbor: int
    
    Returns
    ----------
    X_new : ndarray, shape (n_samples_new, n_features)
            Synthetically generated samples.
    y_new : ndarray, shape (n_samples_new,)
            Target values for synthetic samples.
    
    """   
    n_samples, n_labels = y.shape
    
    # computer k-nn
    x, y = shuffle(x, y)
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(x)
    
    x_new, y_new = x, y
    # search which class is majority or minority
    j = 0
    while True:
        label_count = np.sum(y_new, axis=0)
        print("step-{}: num_majority / num_minority: {}".format(j, np.max(label_count)/np.min(label_count)))
        if np.max(label_count)/np.min(label_count) < criterion_rate:
            break
        
        m = np.argmin(label_count)
        indx = (y[:,m] == 1)
        minority_x, minority_y = x[indx], y[indx]
        nn = neigh.kneighbors(minority_x, k_neighbors)
        for  neighIndx in nn[1]:
            i = random.randint(1,k_neighbors-1)

            if np.all( y[neighIndx[0]]==y[neighIndx[i]]):
                r = random.random()
                temp_x = [(1-r) * x[neighIndx[0]] + r * x[neighIndx[i]]]
                temp_y = [y[neighIndx[0]]]
            else:
                c = np.sum( (y[neighIndx[0]]==y[neighIndx[i]]).astype(float))/n_labels
                r = random.random()*c
                temp_x = [(1-r) * x[neighIndx[0]] + r * x[neighIndx[i]]]
                temp_y = (1-r) * y[neighIndx[0]] + r * y[neighIndx[i]]
                temp_y = [(temp_y > 0.5).astype(float)]
                
            temp_x = np.array(temp_x)
            temp_y = np.array(temp_y)
            x_new = np.concatenate((x_new, temp_x))
            y_new = np.concatenate((y_new, temp_y))
        
        j += 1
        if j > maxstep:
            break
        
    x_new, y_new = shuffle(x_new, y_new)
    return x_new, y_new

def singleLabelSMOTE(X, k_neighbors=5, rate=3):  
    """
    Synthetic Minority Over-sampling Technique on single-label dataset

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        
    k_neighbors : int
        
    rate : float

    Returns
    -------
    X_new : ndarray
        
    """
    N = X.shape[0]
    N_new = int(N*rate)
    
    # computer k-nn
    X = shuffle(X)
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X)
    
    X_new = X
    
    j = 0
    
    for x in X:
        knn = neigh.kneighbors(x.reshape(1,-1), k_neighbors)
        for _ in range(int(np.floor(rate))):
            m = random.randint(1,k_neighbors-1)
            r = random.random()
            k = knn[1][0,m]
            sx = [(1-r) * x + r * X[k]]
            X_new = np.concatenate((X_new, sx))
            j += 1
            
    if j >= N_new:
        return X_new
    else:
        X = shuffle(X)      
        for x in X:
            knn = neigh.kneighbors(x.reshape(1,-1), k_neighbors)
            m = random.randint(1,k_neighbors-1)
            r = random.random()
            k = knn[1][0,m]
            sx = [(1-r) * x + r * X[k]]
            X_new = np.concatenate((X_new, sx))
            j += 1   
            if j >= N_new:
               break
        return X_new

if __name__ == "__main__":
    data = np.load('mlecFeatures.npz')  
    x,y=data['x'],data['y']  
    x_new, y_new = MLSMOTE(x,y)