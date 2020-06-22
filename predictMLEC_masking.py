# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:47:49 2020

@author: lwzjc
"""

from Bio import SeqIO
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from prepareDataset import readMLEnzyme
from aaindexValues import aaindex1PCAValues
from sklearn.model_selection import KFold

def subsetAcc(y_true, y_pred):
    account = 0
    n = y_true.shape[0]
    for i in range(n):
        for j in range(7):
            if y_true[i,j] != y_pred[i,j]:
                account += 1
                break
    return (n-account)/n

def hamming_loss(y_true, y_pred):
    account = 0
    n = y_true.shape[0]
    for i in range(n):
        for j in range(7):
            if y_true[i,j] != y_pred[i,j]:
                account += 1
    return account/(n*7)

def load_mlec(nr=80, n_features=15):
    aadict = aaindex1PCAValues(n_features)
    mlec_seqs, mlec_labels = readMLEnzyme()
    x = np.zeros(shape=(3156, 7740, n_features))
    labels = []
    i = 0
    for seq_record in SeqIO.parse('data/mlec_80.fasta', 'fasta'):
        s = seq_record.id
        pid = s.split(' ')
        protId = pid[0]
        
        j = 0
        for aa in mlec_seqs[protId]:
            if aa not in ['X','B','Z']:
                x[i,j,:] = aadict[aa]
            j += 1
            
        labels.append(mlec_labels[protId])
        
        i += 1
    
    y = np.array(labels)
    
    return x, y

def attention_block(inputs, TIME_STEPS):
    # inputs.shape=(batch_size, time_steps, lstm_units)
    a = layers.Permute((2,1))(inputs)
    a = layers.Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = layers.Permute((2,1), name='attention_vec')(a)
    output_attention = layers.Multiply()([inputs, a_probs])
    return output_attention

def get_attention_model(TIME_STEPS, INPUT_DIM):
    inputs = layers.Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    mask_inputs = layers.Masking(mask_value=0)(inputs)
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(mask_inputs)
    attention_mul = attention_block(lstm_out, TIME_STEPS)
    attention_mul = layers.Flatten()(attention_mul)
    output = layers.Dense(7, activation='sigmoid')(attention_mul)
    model = Model(inputs=inputs, outputs=output)
    return model

def masking_model(n_features):
    model = Sequential()
    model.add( layers.Masking(mask_value=0, input_shape=[7740, n_features]))
    model.add( layers.Bidirectional(layers.LSTM(10)))
    model.add( layers.Dense(7, activation='sigmoid'))
    model.summary()
    
    return model

lr = 0.001
TIME_STEPS = 7740
INPUT_DIM = 15
x, y = load_mlec(nr=80, n_features=INPUT_DIM)

y_pred = np.zeros((0, 7))
y_true = np.zeros((0, 7))

k = 0    
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(x,y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    tf.keras.backend.clear_session()
    
    #model = masking_model(INPUT_DIM)
    model = get_attention_model(TIME_STEPS, INPUT_DIM)
    modelfile = './model/mlec/weights-mlec80-mask-{}.h5'.format(k)
   
    model.compile(optimizer=Adam(learning_rate=lr),
         loss='binary_crossentropy',
         metrics=['accuracy'])
    '''print(model.summary())
    lr_decay = LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))

    checkpoint = ModelCheckpoint(modelfile, monitor='val_loss',
                                   save_best_only=True, 
                                   save_weights_only=True, 
                                   verbose=1)   
    model.fit(x_train, y_train,
              batch_size=64,
              epochs=20,
              validation_data=[x_test, y_test],
              callbacks=[checkpoint, lr_decay])
    '''
    model.load_weights(modelfile)
    pred = model.predict(x_test)
    
    y_pred = np.concatenate((y_pred, pred))
    y_true = np.concatenate((y_true,y_test))
    
    k += 1
    
c = np.sum(y_true,axis=0)/(len(y_true))
#c=0.5
y_p = (y_pred>c).astype(float)
info = '\n\n--- use melc_80 dataset, data with mask ---\n'

with open('ml_result.txt', 'a') as fw:
    fw.write(info)
    fw.write("hamming loass = {}\n".format(hamming_loss(y_true, y_p)))
    fw.write("subset accuracy = {}\n".format( subsetAcc(y_true, y_p)))




