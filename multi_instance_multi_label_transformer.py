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
from sklearn import metrics
import scipy.io as sio
from nlp_transformer import Encoder, create_padding_mask
from prepare_seq import multi_instances_split, protseq_to_vec
from tools import displayMLMetrics

def load_mlec_nr(nrfile='data/mlec_40.fasta', npzfile='mlec_nr40.npz', description=False, firstly_load=False):
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
        
        np.savez(npzfile, seqs=seqs, labels=labels)
    else:
        data = np.load(npzfile, allow_pickle=True)
        seqs, labels = data['seqs'], data['labels']
    
    return seqs, labels

class MILSTMModel(Model):
    def __init__(self, maxlen=100, num_fragment=5):
        super(MILSTMModel, self).__init__()
        self.num_fragment = num_fragment
        self.maxlen = maxlen
        self.embd = layers.Embedding(24, 10)
        self.brnn = layers.Bidirectional(layers.LSTM(10))
        self.d1 = layers.Dense(64, activation='relu')
        self.dp = layers.Dropout(0.5)
        self.d2 = layers.Dense(7, activation='sigmoid')
    def call(self, seq):
        seq_frags = multi_instances_split(seq, num_fragment=self.num_fragment)
        
        mout = []
        for j in range(self.num_fragment):
            t = []
            for i in range(len(seq)):
                t.append(seq_frags[i][j])
            x = protseq_to_vec(t, maxlen=self.maxlen)
            x = self.embd(x)
            x = self.brnn(x)
            x = self.d1(x)
            x = self.dp(x)
            x = self.d2(x)
            
            mout.append(x)
        out = layers.Maximum()(mout)    
        out = layers.Dense(7, activation='sigmoid')(out)
        return out
        
class MITransformerModel(Model):
    def __init__(self, n_layers=4, embed_dim=8, num_heads=2, ff_dim=64,
                 vocab_size=22, maxlen=100, droprate=0.2, num_fragment=5):
        super(MITransformerModel, self).__init__()
        #self.padding_mask = create_padding_mask()
        self.encoder = Encoder(n_layers=n_layers, d_model=embed_dim, n_heads=num_heads, ffd=ff_dim,
            input_vocab_size=vocab_size, max_seq_len=maxlen, dropout_rate=droprate)
        self.gp = layers.GlobalAveragePooling1D()
        self.dp = layers.Dropout(0.2)
        self.d1 = layers.Dense(128, activation="relu")
        self.d2 = layers.Dense(7, activation="sigmoid")
        self.d3 = layers.Dense(7, activation="sigmoid")
        self.maxlen = maxlen
        self.num_fragment = num_fragment
    def call(self, seq):
        seq_frags = multi_instances_split(seq, num_fragment=self.num_fragment)
        flag = True
        
        for j in range(self.num_fragment):
            t = []
            for i in range(len(seq)):
                t.append(seq_frags[i][j])
            
            x = protseq_to_vec(t, maxlen=self.maxlen)
            mask = create_padding_mask(x)
            x = self.encoder(x, True, mask)
            x = self.gp(x)
            x = self.dp(x)
            x = self.d1(x)
            x = self.d2(x)
            if flag:
                out = x
                flag = False
            else:
                out = layers.Concatenate()([out, x])
        
        out = self.d3(out)
        return out

def loss(model, x, y):
    y_ = model(x)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return loss_object(y_true=y, y_pred=y_)  
  
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train(model, dataset, maxlen=100, num_epochs=20):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_loss_results = []
    train_accuracy_results = []
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        for seq, y in dataset:
            loss_value, grads = grad(model, seq, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            epoch_loss_avg(loss_value)
            epoch_accuracy(y, model(seq))
        
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        
        if epoch % 2 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
        
#seqs, labels = load_mlec_nr(firstly_load=False)
seqs, labels = load_mlec_nr(nrfile='data/mlec_90.fasta', 
                            npzfile='mlec_nr90.npz', firstly_load=False)
labels = np.array(labels)  
N = len(seqs)
y_pred = np.ndarray(shape=(0,7))
y_true = np.ndarray(shape=(0,7))
for i in range(N):
    train_index = [True]*N
    train_index[i] = False
    train_seqs, train_labels = seqs[train_index], labels[train_index]
    test_seqs, test_labels = [seqs[i]], labels[i]
    test_seqs = np.array(test_seqs)
    test_labels = test_labels[np.newaxis,:]
    
    dataset = tf.data.Dataset.from_tensor_slices((train_seqs, train_labels))
    dataset = dataset.shuffle(100).batch(32)

    #model = MITransformerModel()
    model = MILSTMModel(maxlen=100,num_fragment=8)
    
    train(model, dataset)
    y_ = model(test_seqs)
    y_pred = np.concatenate((y_pred, y_))
    y_ture = np.concatenate((y_true, test_labels))
    print("{} predicted:{}".format(i,y_pred))
info = "predict ML-Enzyme by Multi-Instance transfomer"
displayMLMetrics(y_true, y_pred, "result", info)    