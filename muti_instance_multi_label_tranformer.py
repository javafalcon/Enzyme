# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:07:48 2020

@author: lwzjc
"""
import numpy as np
import re
from prepareDataset import readMLEnzyme, load_MIML
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from skmultilearn.model_selection import IterativeStratification
from sklearn import metrics
import scipy.io as sio


def load_mlec_nr(nrfile='data/melc_40.fasta', npzfile='melc_nr40.npz', description=False, firstly_load=False):
    if firstly_load:
        mlec_seqs, mlec_labels = readMLEnzyme()
        seqs, labels = [], []
        
        for seq_record in SeqIO.parse(nrfile, 'fasta'):
            s = seq_record.id
            if description:
                pid = s.split('|')
                protId = pid[1]
            else:
                pid = s.split(' ')
                protId = pid[0]
            seqs.append(mlec_seqs[protId])
            labels.append(mlec_labels[protId])
        
        np.save(npzfile, seqs=seqs, labels=labels)
    else:
        data = np.load(npzfile, allow_pickle=True)
        seqs, labels = data['seqs'], data['labels']
    
    return seqs, labels

def model(embed_dim=16, num_heads=4, ff_dim=128, vocab_size=21, maxlen=100):
    inputs = layers.Input(shape=(maxlen,))

    encode_padding_mask = create_padding_mask(inputs)
    encoder = Encoder(n_layers=4, d_model=embed_dim, n_heads=num_heads, ffd=ff_dim,
            input_vocab_size=vocab_size, max_seq_len=maxlen, dropout_rate=0.2)
    x = encoder(inputs, False, encode_padding_mask)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(7, activation="sigmoid")(x)