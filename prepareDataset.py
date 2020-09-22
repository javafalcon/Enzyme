# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:12:44 2020

@author: Administrator
"""
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import numpy as np
from SeqFormulate import cor_chaosGraph, DAA_chaosGraph, corr_onehot
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import random
nr40 = ['data/slec_{}_40.fasta'.format(i) for i in range(1,8)]
nr60 = ['data/slec_{}_60.fasta'.format(i) for i in range(1,8)]
nr80 = ['data/slec_{}_80.fasta'.format(i) for i in range(1,8)]
nr100 = ['data/slec_{}.fasta'.format(i) for i in range(1,8)]


def readAllEnzymeSeqsML():
    files=['ec_1.fasta', 'ec_2.fasta', 'ec_3.fasta', 'ec_4.fasta', 'ec_5.fasta',
           'ec_6.fasta', 'ec_7.fasta']
    data={}
    target={}
    for i in range(7):
        file = os.path.join('data', files[i])
        for seq_record in SeqIO.parse(file, 'fasta'):
            if len(str(seq_record.seq)) < 50:
                continue
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

def writeSLEC(data, target):
    MLEC_seqs, MLEC_labels = {}, {}
    SLEC_seqs, SLEC_labels = {}, {}
    
    for key in data.keys():
        targ = target[key]
        if np.sum(targ) > 1: # multi-label
            MLEC_seqs[key] = data[key]
            MLEC_labels[key] = target[key]
        else: # single-label
            SLEC_seqs[key] = data[key]
            SLEC_labels[key] = np.argmax(targ)
            
    # write into files
    seq_records_ls = [[] for i in range(7)]
    for key in SLEC_seqs.keys():
        seq_record = SeqRecord(Seq(SLEC_seqs[key], IUPAC.protein), id=key)
        seq_records_ls[SLEC_labels[key]].append(seq_record)
    
    for i in range(7):
        SeqIO.write(seq_records_ls[i],'data/slec_{}.fasta'.format(i+1),'fasta')
            
# read nr40 data as single label
# Proteins who have multi lables were removed
def readSLEnzyme(files):
    prot_seqs = {}
    prot_labels = {}

    for i in range(len(files)):
         for seq_record in SeqIO.parse(files[i], 'fasta'):
            if len(str(seq_record.seq)) < 50:
                continue
            seqid = seq_record.id
            seqid = seqid.split(' ')
            seqid = seqid[0]

            if seqid in prot_seqs.keys():
                # The protein has multi-lables, so remove it from protein dict
                prot_seqs.pop(seqid)
                prot_labels.pop(seqid)
            else:
                # The protien is firstly seen, so add it to protein dict               
                prot_seqs[seqid] = str(seq_record.seq)
                prot_labels[seqid] = i
    
    return prot_seqs, prot_labels

def readMLEnzyme():
    '''files=['ec_1_40.fasta', 'ec_2_40.fasta', 'ec_3_40.fasta', 'ec_4_40.fasta',
           'ec_5_40.fasta', 'ec_6_40.fasta', 'ec_7_40.fasta']'''
    files=['ec_1.fasta', 'ec_2.fasta', 'ec_3.fasta', 'ec_4.fasta', 'ec_5.fasta',
           'ec_6.fasta', 'ec_7.fasta']
    
    prot_seqs = {}
    prot_labels = {}
    MLEC_seqs = {}
    MLEC_labels = {}
    
    for i in range(7):
        file = os.path.join('data', files[i])
        for seq_record in SeqIO.parse(file, 'fasta'):
            if len(str(seq_record.seq)) < 50:
                continue
            seqid = seq_record.id
            seqid = seqid.split('|')
            seqid = seqid[1]

            if seqid in prot_seqs.keys():
                # The protein has multi-lables, so add its label
                MLEC_seqs[seqid] = str(seq_record.seq)
                t = MLEC_labels.get(seqid, prot_labels[seqid])
                t[i] = 1
                MLEC_labels[seqid] = t
                
            else:
                # The protien is firstly seen, so add it to protein dict               
                prot_seqs[seqid] = str(seq_record.seq)
                t = np.zeros((7,))
                t[i] = 1
                prot_labels[seqid] = t
    return MLEC_seqs, MLEC_labels

def getNotEnzyme(n_samples, random_state=None):
    seq_records = SeqIO.parse('data/NotEC_40.fasta', 'fasta')
    seq_records = shuffle(list(seq_records), random_state=random_state)
    
    n = 0
    prot_seqs = []
    for seq_record in seq_records:
        if len(str(seq_record.seq)) >= 50:
            prot_seqs.append(str(seq_record.seq))
            n += 1
        if n > n_samples:
            break
    return prot_seqs

def load_SL_EC_data(files):
    prot_seqs, prot_labels = readSLEnzyme(files)
    seqs, labels = [], []
    for key in prot_seqs.keys():
        seqs.append(prot_seqs[key])
        labels.append(prot_labels[key])
    X = DAA_chaosGraph(seqs)
    y = np.array(labels)
    return X, y

def load_ML_SL_EC_data():
    mlec_seqs = []
    for seq_record in SeqIO.parse('data\\mlec_60.fasta','fasta'):
        mlec_seqs.append(str(seq_record.seq))
    mlec_x = DAA_chaosGraph(mlec_seqs)

    slec_seqs, _ = readSLEnzyme(nr40)
    slec_x = DAA_chaosGraph(slec_seqs)
    slec_x = shuffle(slec_x)
    x = np.concatenate((slec_x[:5000], mlec_x))  
    y = np.ones((len(x),))
    y[:5000] = 0
    
    return x, y

def load_mlec_nr(nr=80):
    mlec_seqs, mlec_labels = readMLEnzyme()
    seqs, labels = [], []
    
    for seq_record in SeqIO.parse('data/mlec_{}.fasta'.format(nr), 'fasta'):
        s = seq_record.id
        pid = s.split(' ')
        protId = pid[0]
        seqs.append(mlec_seqs[protId])
        labels.append(mlec_labels[protId])
    
    x = np.ndarray(shape=(len(seqs), 21, 21, 1))    
    x[:,:,:,0] = DAA_chaosGraph(seqs)
    
    y = np.array(labels)
    return x, y

def gen_bag_inst(seq, k, num_inst=10):
    """
    生成num_inst个长度为k的蛋白质序列片段

    Parameters
    ----------
    seq : str
        蛋白质序列.
    k : int
        子序列片段的长度.
    num_inst : int
        子序列的个数.

    Returns
    -------
    list.

    """
    seqs = []
    if len(seq) < k:
        for _ in range(num_inst):
            seqs.append(seq)
        
    else:
        seqs.append( seq[:k])
        seqs.append( seq[-k:])
        w = len(seq) - k
        for _ in range(num_inst-2):
            start = random.randint(0, w)
            seqs.append( seq[start:start+k])
    return seqs      
    
def load_MIML(nr=90, fragLen=100, num_inst=10):
    mlec_seqs, mlec_labels = readMLEnzyme()
    seq_bags, seq_labels = [], []
    
    X = []
    for seq_record in SeqIO.parse('data/mlec_{}.fasta'.format(nr), 'fasta'):
        s = seq_record.id
        pid = s.split(' ')
        protId = pid[0]
         
        seq_bags = gen_bag_inst(mlec_seqs[protId], k=fragLen, num_inst=num_inst)
        x = DAA_chaosGraph(seq_bags)
        
        X.append(x)
        seq_labels.append(mlec_labels[protId])
        
    X = np.array(X)
    y = np.array(seq_labels)
    
    return X, y

def load_MISL(files, fragLen=100, num_inst=10):
    prot_seqs, prot_labels = readSLEnzyme(files)
    seq_bags, seq_labels = [], []
    X = []
    for key in prot_seqs.keys():
        seq_bags = gen_bag_inst(prot_seqs[key], k=fragLen, num_inst=num_inst)
        x = DAA_chaosGraph(seq_bags)
        
        X.append(x)
        seq_labels.append(prot_labels[key])
        
    X = np.array(X)
    y = np.array(seq_labels)
    
    return X, y

def load_data(vers=1, r=1):
    if vers == 1:
        ec_seqs, ec_labels = readSLEnzyme(nr40)
        not_ec = getNotEnzyme(27907, random_state=42)
        pos_x = DAA_chaosGraph(list(ec_seqs.values()))
        neg_x = DAA_chaosGraph(not_ec)
        x = np.concatenate((pos_x, neg_x))
        y = np.zeros((len(x),))
        y[:len(pos_x)] = 1
        x, y = shuffle(x, y)
        return x, y
    elif vers == 2:
        seq_records = SeqIO.parse('data/EC_40.fasta', 'fasta')
        seq_records = shuffle(list(seq_records), random_state=42)
        Enzyme_seqs = []
        for seq_record in seq_records:
            if 5000 >= len(str(seq_record.seq)) >= 50 :
                Enzyme_seqs.append(str(seq_record.seq))
              
        seq_records = SeqIO.parse('data/NotEC_40.fasta', 'fasta')
        seq_records = shuffle(list(seq_records), random_state=42)
        notEnzyme_seqs = []
        for seq_record in seq_records:
            if 5000 >= len(str(seq_record.seq)) >= 50:
                notEnzyme_seqs.append(str(seq_record.seq))
        notEnzyme_seqs = notEnzyme_seqs[:len(Enzyme_seqs)]
        
        seqs = Enzyme_seqs + notEnzyme_seqs
        labels = [1 for i in range(len(Enzyme_seqs))] + [0 for i in range(len(notEnzyme_seqs))]
        x = corr_onehot(seqs,r)
        y = np.array(labels)
        #pos_x = DAA_chaosGraph(Enzyme_seqs)
        #neg_x = DAA_chaosGraph(notEnzyme_seqs)
        x, y = shuffle(x, y)
        
        return x, y

def load_Kf_data(X, y, kfold=5, random_state=None):
    X_train_Kf, y_train_Kf = [], []
    X_test_Kf, y_test_Kf = [], []

    skf = StratifiedKFold(n_splits=kfold, random_state=random_state, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train_Kf.append(X_train)
        y_train_Kf.append(y_train)
        
        X_test_Kf.append(X_test)
        y_test_Kf.append(y_test)
        
    return (X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf)

def load_Kf_data_with_weight(X, y, weight, kfold=5, random_state=None):
    X_train_Kf, y_train_Kf = [], []
    X_test_Kf, y_test_Kf = [], []
    weight_Kf = []
    skf = StratifiedKFold(n_splits=kfold, random_state=random_state, shuffle=True)
    for train_index, test_index in skf.split(X, y, weight):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weight_train = weight[train_index]
        X_train_Kf.append(X_train)
        y_train_Kf.append(y_train)
        
        weight_Kf.append(weight_train)
        
        X_test_Kf.append(X_test)
        y_test_Kf.append(y_test)
        
    return (X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf), weight_Kf

def statInfo():
    seq_records = SeqIO.parse('data/EC_40.fasta', 'fasta')
    Enzyme_seqs = []
    for seq_record in seq_records:
        if len(str(seq_record.seq)) >= 50:
            Enzyme_seqs.append(str(seq_record.seq))
            
    leninfo = {}
    for seq in Enzyme_seqs:
        k = len(seq)//100
        leninfo[ k] = leninfo.get(k, 0) + 1
        
    return leninfo
   
if __name__ == "__main__": 
    """
    with open('data/2.txt', 'r') as fr:
        # lines[0]: Enzymes; lines[1]: Non enzyme
        lines = fr.readlines()
    data, target = readAllEnzymeSeqsML()
    protAccs = lines[1].split()
    enzymes = {}
    count = 0
    for i in range(3, len(protAccs)):
        key = protAccs[i]
        if key in data.keys():
            enzymes[key] = target[key]
        else:
            count += 1
            print("\rnot enzymes: {}".format(count))
    """
    '''
    from Bio import ExPASy
    from Bio import SwissProt
    accession = 'Q5XIZ6'
    handle=ExPASy.get_sprot_raw(accession) 
    record=SwissProt.parse(handle)
    try:
        record = SwissProt.read(handle)
    except:
        print("WARNING: Accession %s not found" % accession)
    '''
    
    #statInfo()
    #data, target = readAllEnzymeSeqsML()
    #writeSLEC(data, target)
    #x,y = load_SL_EC_data()
    '''
    # stastic number of proteins with multi-label
    MLECseqs, MLEClabels = readMLEnzyme()
    keys = MLECseqs.keys()
    for seq_record in SeqIO.parse('data\\pos_all_90.fasta','fasta'):
        seqid = seq_record.id
        seqid = seqid.split('|')
        seqid = seqid[1]
        if seqid not in keys :
            print("{} has not multi-label".format(seqid))
    '''
    
    '''
    MLECseqs, MLEClabels = readMLEnzyme()
    lenstat = {}
    for key in MLECseqs.keys():
        seq = MLECseqs[key]
        lenstat[len(seq)] = lenstat.get(len(seq), 0) + 1
        
    items = list(lenstat.items())
    items.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(10):
        item = items[i]
        print("{0:<6}{1:>5}".format(item[0],item[1]))
    '''
    
    '''
    seq_records = []
    for key in MLECseqs.keys():
        seq_record = SeqRecord(Seq(MLECseqs[key], IUPAC.protein), id=key)
        seq_records.append(seq_record)
    SeqIO.write(seq_records,'mlec.fasta','fasta')
    '''
    
    '''    
    count = 0
    for key in ECseqs.keys():
        if len(ECseqs[key]) < 50:
            count += 1
            print("{}:{}'s length is less than 50".format(count,key))
    '''
    
    '''
    # 统计各个长度段序列数
    items.sort(key=lambda x:x[0]) 
    statlen = np.zeros((7,))
    for item in items:
        if item[0] < 1000:
            statlen[0] += 1
        elif item[0] < 2000:
            statlen[1] += 1
        elif item[0] < 3000:
            statlen[2] += 1
        elif item[0] < 4000:
            statlen[3] += 1
        elif item[0] < 5000:
            statlen[4] += 1
        elif item[0] < 6000:
            statlen[5] += 1
        else:
            statlen[6] += 1
    '''
    
    leninfo = statInfo()
    lenitems = leninfo.items()
    sorted(lenitems, key=lambda x: x[0])
    
    
        