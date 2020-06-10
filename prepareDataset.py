# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:12:44 2020

@author: Administrator
"""
import os
from Bio import SeqIO
import numpy as np

def readAllSeqsML():
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
                data[protId] = seq_record.seq
                target[protId] = t
    
    return data, target

def readNr40SL():
    files=['ec_1_40.fasta', 'ec_2_40.fasta', 'ec_3_40.fasta', 'ec_4_40.fasta',
           'ec_5_40.fasta', 'ec_6_40.fasta', 'ec_7_40.fasta']
    '''files=['ec_1.fasta', 'ec_2.fasta', 'ec_3.fasta', 'ec_4.fasta', 'ec_5.fasta',
           'ec_6.fasta', 'ec_7.fasta']'''
    prots={}
    labels={}
    
    for i in range(7):
        file = os.path.join('data', files[i])
        for seq_record in SeqIO.parse(file, 'fasta'):
            seqid = seq_record.id
            seqid = seqid.split('|')
            seqid = seqid[1]
            if seqid in prots.keys(): # seqid has been seen in other file
                prots.pop(seqid)
                labels.pop(seqid)
            else:
                prots[seqid] = seq_record.seq
                labels[seqid] = i
    
    return prots, labels
    
if __name__ == "__main__":   
    prots, labels = readNr40SL()       
        
        