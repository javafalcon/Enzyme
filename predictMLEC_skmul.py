# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:42:57 2020

@author: lwzjc
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:20:26 2020

@author: Administrator
"""

from SeqFormulate import DAA_chaosGraph
import numpy as np
import re
from prepareDataset import readMLEnzyme
from Bio import SeqIO
from sklearn.utils import shuffle
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold

def kneighbors(x, X, k, m=0):
    """
    Parameters
    ----------
    x : 1-by-m 向量
        表示一个样本.
    X : n-by-m矩阵，表示n个样本，每个样本是m维向量
    k : int，邻居数
    m : 从第m个邻居开始
    Returns
    -------
    X中与x最近的样本序号，按两个向量的cos夹角距离比较.

    """
    dist = np.zeros(shape=(X.shape[0],))
    for i in range(X.shape[0]):
        dist[i] = np.dot(x, X[i])/(np.sqrt(np.dot(x,x))*np.sqrt(np.dot(X[i],X[i])))
    knn_indx = np.argsort(dist)  
    knn_indx = knn_indx[::-1]
    
    return knn_indx[m:k+m]

def knn_feature(x, X, Y, k, m=0):
    """
    Parameters
    ----------
    x : 1-by-m向量，表示一个样本.
    X : n-by-m矩阵，表示n个样本，每个样本是m维向量
    Y : X的标签向量，n-by-L矩阵，没行表示一个样本的多标签的one-hot表示
    k : int，邻居数.

    Returns
    -------
    1-by-L的向量f，f(i)表示x的k个邻居有标签i的比例

    """
    knn_indx = kneighbors(x,X,k,m)
    a = np.zeros((Y.shape[1]))
    for i in range(k):
        a += Y[knn_indx[i]]
    a /= k
    return a

def knn_feature_express(X_train, Y_train, X_test=None, k=5):
    if X_test is not None:
        v = np.ndarray((X_test.shape[0], Y_train.shape[1]))
        for i in range(X_test.shape[0]):
            v[i,:] = knn_feature(X_test[i], X_train, Y_train, k)
    else:
        v = np.ndarray((X_train.shape[0], Y_train.shape[1]))
        for i in range(X_train.shape[0]):
            v[i,:] = knn_feature(X_train[i], X_train, Y_train, k,m=1)
    return v
        
def daa(seqs):
    AminoAcids = 'ARNDCQEGHILKMFPSTWYVX'
    x = []
    regexp = re.compile('[^ARNDCQEGHILKMFPSTWYVX]')
    
    for seq in seqs:
        seq = regexp.sub('X', seq)
        t = np.zeros((21,21))
        for i in range(len(seq)-1):
            t[AminoAcids.index(seq[i])][AminoAcids.index(seq[i+1])] += 1
        x.append(t/np.sum(t))
    return np.array(x)

def load_mlec():
    mlec_seqs, mlec_labels = readMLEnzyme()
    seqs, labels = [], []
    
    for key in mlec_seqs.keys():
        seqs.append(mlec_seqs[key])
        labels.append(mlec_labels[key])
        
    x = np.ndarray(shape=(len(seqs), 21, 21, 2))    
    x[:,:,:,0] = DAA_chaosGraph(seqs)
    x[:,:,:,1] = daa(seqs)
    y = np.array(labels)
    return x, y

def load_mlec_nr(nr=80):
    mlec_seqs, mlec_labels = readMLEnzyme()
    seqs, labels = [], []
    #x = []
    weight = np.ones((22,))
    weight[20] = 1000
    weight[21] = 0.1
    from SeqFormulate import greyPseAAC
    for seq_record in SeqIO.parse('data/mlec_{}.fasta'.format(nr), 'fasta'):
        s = seq_record.id
        pid = s.split(' ')
        protId = pid[0]
        seq = mlec_seqs[protId]
        seqs.append(seq)
        labels.append(mlec_labels[protId])
        #x.append( greyPseAAC(seq, ["Hydrophobicity"], weight=weight, model=1))
    #x = np.ndarray(shape=(len(seqs), 21, 21))    
    #x[:,:,:,0] = DAA_chaosGraph(seqs)
    #x[:,:,:,1] = daa(seqs)
    x = daa(seqs)
    x = np.array(x)
    y = np.array(labels)
    return x, y
        
def load_mlec_v2():
    mlec_seqs, mlec_labels = readMLEnzyme()
    seqs, labels = [], []
    for key in mlec_seqs.keys():
        seqs.append(mlec_seqs[key])
        labels.append(mlec_labels[key])
    x = DAA_chaosGraph(seqs)
    y = np.array(labels)
    return x, y


if __name__ == "__main__":
    #result = resnetWithAttention_main()
    x, y = load_mlec_nr(nr=90)
    x, y = shuffle(x, y)
    x = x.reshape(-1, 441)
    from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
    graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
    edge_map = graph_builder.transform(y)
    print("{} labels, {} edges".format(y.shape[1], len(edge_map)))
    
    from skmultilearn.cluster.networkx import NetworkXLabelGraphClusterer
    from skmultilearn.ensemble import LabelSpacePartitioningClassifier
    
    clusterer = NetworkXLabelGraphClusterer(graph_builder, method='label_propagation')
    clf = LabelSpacePartitioningClassifier(
        classifier=LabelPowerset(classifier=RandomForestClassifier(n_estimators=100), 
                             require_dense=[True,True]),
        require_dense=[True, True],
        clusterer=clusterer)
    """
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_knn_feature = knn_feature_express(X_train, y_train)
        X_test_knn_feature = knn_feature_express(X_train, y_train, X_test)
        
        X_train = np.hstack((X_train, X_train_knn_feature))
        X_test = np.hstack((X_test, X_test_knn_feature))
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
    """
    """
    x_knn_feature = knn_feature_express(x, y, k=7)
    X = np.hstack((x, x_knn_feature))
    Xsum = np.sum(X,axis=1)
    newX = np.ndarray(shape=X.shape)
    
    for i in range(X.shape[0]):
        newX[i] = X[i]/Xsum[i]
    """
    result = cross_validate(clf, x, y, cv=5, scoring="accuracy", return_train_score=True)

    #statlen = statInfo()
    """
    KERAS_PARAMS = dict(epochs=10, batch_size=100, verbose=0)
    from skmultilearn.dataset import load_dataset

    X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
    X_test, y_test, _, _ = load_dataset('emotions', 'test')
    clf = BinaryRelevance(classifier=Keras(create_model_single_class, False, KERAS_PARAMS), require_dense=[True,True])
    clf.fit(X_train, y_train)
    result = clf.predict(X_test)
    """