# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:12:44 2020

@author: Administrator
"""
import os
from Bio import SeqIO
import numpy as np
from SeqFormulate import DAA_chaosGraph
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

def readAllEnzymeSeqsML():
    files=['ec_1.fasta', 'ec_2.fasta', 'ec_3.fasta', 'ec_4.fasta', 'ec_5.fasta',
           'ec_6.fasta', 'ec_7.fasta']
    data={}
    target={}
    for i in range(7):
        file = os.path.join('data', files[i])
        for seq_record in SeqIO.parse(file, 'fasta'):
            s = seq_record.id
            pid = s.split('|')
            protId = pid[1]
            
            t = target.get(protId, np.zeros(shape=(7,)))
            t[i] = 1
            if protId in data.keys():
                target[protId] = t
            else:
                data[protId] = str(seq_record.seq)
                target[protId] = t
    
    return data, target


# read nr40 data as single label
# Proteins who have multi lables were removed
def readEnzymeNr40SL():
    files=['ec_1_40.fasta', 'ec_2_40.fasta', 'ec_3_40.fasta', 'ec_4_40.fasta',
           'ec_5_40.fasta', 'ec_6_40.fasta', 'ec_7_40.fasta']
    '''files=['ec_1.fasta', 'ec_2.fasta', 'ec_3.fasta', 'ec_4.fasta', 'ec_5.fasta',
           'ec_6.fasta', 'ec_7.fasta']'''
    prot_seqs = {}
    prot_labels = {}

    for i in range(7):
        file = os.path.join('data', files[i])
        for seq_record in SeqIO.parse(file, 'fasta'):
            seqid = seq_record.id
            seqid = seqid.split('|')
            seqid = seqid[1]

            if seqid in prot_seqs.keys():
                # The protein has multi-lables, so remove it from protein dict
                prot_seqs.pop(seqid)
                prot_labels.pop(seqid)
            else:
                # The protien is firstly seen, so add it to protein dict
                prot_seqs[seqid] = str(seq_record.seq)
                prot_labels[seqid] = i
    
    return prot_seqs, prot_labels

def getNotEnzyme(n_samples, random_state=None):
    seq_records = SeqIO.parse('data/NotEC_40.fasta', 'fasta')
    seq_records = shuffle(list(seq_records), random_state=random_state)
    
    n = 0
    prot_seqs = []
    for seq_record in seq_records:
        prot_seqs.append(str(seq_record.seq))
        n += 1
        if n > n_samples:
            break
    return prot_seqs

def load_EC_data():
    prot_seqs, prot_labels = readEnzymeNr40SL()
    seqs, labels = [], []
    for key in prot_seqs.keys():
        seqs.append(prot_seqs[key])
        labels.append(prot_labels[key])
    X = DAA_chaosGraph(seqs)
    y = np.array(labels)
    return X, y

def load_data():
    ec_seqs, ec_labels = readEnzymeNr40SL()
    not_ec = getNotEnzyme(27909, random_state=42)
    pos_x = DAA_chaosGraph(ec_seqs)
    neg_x = DAA_chaosGraph(not_ec)
    x = np.concatenate((pos_x, neg_x))
    y = np.zeros((len(x),))
    y[:len(pos_x)] = 1
    return x, y

def load_Kf_data(kfold=5, random_state=None):
    X_train_Kf, y_train_Kf = [], []
    X_test_Kf, y_test_Kf = [], []
    X, y = load_data()
    skf = StratifiedKFold(n_splits=kfold, random_state=random_state, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train_Kf.append(X_train)
        y_train_Kf.append(y_train)
        
        X_test_Kf.append(X_test)
        y_test_Kf.append(y_test)
        
    return (X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf)

if __name__ == "__main__":   
    X, y = load_data()

        
        