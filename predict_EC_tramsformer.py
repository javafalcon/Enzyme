# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 08:26:12 2020

@author: lwzjc
"""
from Bio import SeqIO

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedShuffleSplit, KFold
#from skmultilearn.model_selection import IterativeStratification

import re
import numpy as np

from nlp_transformer import Encoder, create_padding_mask
from prepare_seq import protseq_to_vec
from tools import displayMetrics, displayMLMetrics, plot_history

from tensorflow import keras
from tensorflow.keras import layers, callbacks

def buildModel(maxlen, vocab_size, embed_dim, num_heads, ff_dim, 
               num_blocks, droprate, fl_size, num_classes):
    inputs = layers.Input(shape=(maxlen,))
    
    encode_padding_mask = create_padding_mask(inputs)
    encoder = Encoder(n_layers=num_blocks, d_model=embed_dim, n_heads=num_heads, 
                      ffd=ff_dim, input_vocab_size=vocab_size, 
                      max_seq_len=maxlen, dropout_rate=droprate)
    x = encoder(inputs, False, encode_padding_mask)
    
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(droprate)(x)
    x = layers.Dense(fl_size, activation="relu")(x)
    x = layers.Dropout(droprate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model
    
def load_seqs():
    """
    read Enzyme and not Enzyme sequences, 
    in which every protein sequence is less than 40% similarity with others.

    Returns
    -------
    seqs:
        protein sequences
    labels:
        if 0 for not Enzyme, else 1 for Enzyme

    """
    
    # read Enzyme and not Enzyme sequences 
    seq_records = SeqIO.parse('data/EC_40.fasta', 'fasta')
    seq_records = shuffle(list(seq_records), random_state=42)
    Enzyme_seqs = []
    for seq_record in seq_records:
        if len(str(seq_record.seq)) >= 50:
            Enzyme_seqs.append(str(seq_record.seq))
            
    seq_records = SeqIO.parse('data/NotEC_40.fasta', 'fasta')
    seq_records = shuffle(list(seq_records), random_state=42)
    notEnzyme_seqs = []
    for seq_record in seq_records:
        if len(str(seq_record.seq)) >= 50:
            notEnzyme_seqs.append(str(seq_record.seq))
    notEnzyme_seqs = shuffle(notEnzyme_seqs)
    notEnzyme_seqs = notEnzyme_seqs[:len(Enzyme_seqs)]
    
    
    seqs = Enzyme_seqs + notEnzyme_seqs
    labels = [1 for i in range(len(Enzyme_seqs))] + [0 for i in range(len(notEnzyme_seqs))]

    return seqs, labels

def load_mlec_seqs(k=0, nr=40):
    from prepareDataset import readMLEnzyme
    
    mlec_seqs, mlec_labels = readMLEnzyme()
    seqs, labels = [], []
    if k == 0:
        if nr == 100:
            nrfile = "data\\mlec.fasta"
        else:
            nrfile = "data\\mlec_{}.fasta".format(nr)
        for seq_record in SeqIO.parse(nrfile, 'fasta'):
            s = seq_record.id
            pid = s.split(' ')
            protId = pid[0]
            seqs.append(mlec_seqs[protId])
            labels.append(mlec_labels[protId])
    elif k == 1:
        import scipy.io as sio
        matdata=sio.loadmat('data/4479_label.mat')
        labels = matdata['label']
        seqs = []
        for record in SeqIO.parse('data/4479(0.9).fasta','fasta'):
            seqs.append(str(record.seq))

    return seqs, labels


def transformer_predictor(X_train, y_train, X_test, y_test, modelfile, params):
    keras.backend.clear_session()

    model = buildModel(params['maxlen'], params['vocab_size'], params['embed_dim'], 
                    params['num_heads'], params['ff_dim'],  params['num_blocks'], 
                    params['droprate'], params['fl_size'], params['num_classes'])
    model.summary()

    checkpoint = callbacks.ModelCheckpoint(modelfile, monitor='val_loss',
                                       save_best_only=True, 
                                       save_weights_only=True, 
                                       verbose=1)
    history = model.fit(
        X_train, y_train, 
        batch_size=params['batch_size'], epochs=params['epochs'], 
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
        )

    plot_history(history)

    model.load_weights(modelfile)
    score = model.predict(X_test)
    
    return score

# transformer net params
params = {}
params['vocab_size'] = 24
params['maxlen'] = 800
params['embed_dim'] = 20 # Embedding size for each token
params['num_heads'] = 5  # Number of attention heads
params['ff_dim'] = 128  # Hidden layer size in feed forward network inside transformer
params['num_blocks'] = 8
params['droprate'] = 0.2
params['fl_size'] = 64
params['num_classes'] = 2
params['epochs'] = 10
params['batch_size'] = 32
# load data
seqs, labels = load_seqs()

# split data into train and test
seqs_train, seqs_test, labels_train, labels_test = train_test_split(seqs, labels, 
                                                    test_size=0.3, 
                                                    random_state=42,
                                                    stratify=labels)
X_train = protseq_to_vec(seqs_train, padding_position="post", maxlen=params['maxlen'])
X_test = protseq_to_vec(seqs_test, padding_position="post", maxlen=params['maxlen'])

y_train = keras.utils.to_categorical(labels_train, params['num_classes'])
y_test = keras.utils.to_categorical(labels_test, params['num_classes'])

# training and test
modelfile = './model/ec/ec_trainsformer_{}_{}.h5'.format(params["maxlen"], "pos")
score = transformer_predictor(X_train, y_train, X_test, y_test, modelfile, params)
pred = np.argmax(score, 1)
displayMetrics(np.argmax(y_test, 1), pred)

"""
#sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
y_score = np.zeros((0, params.get('num_classes',2)))
y_true = np.zeros((0, params.get('num_classes',2)))
    
kf = KFold(n_splits=5, shuffle=True)
k = 0
for train_index, test_index in kf.split(seqs, labels):   
#for train_index, test_index in sss.split(seqs, labels):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    k = k+1
    seqs_train, labels_train = [], []
    for i in train_index:
        seqs_train.append(seqs[i])
        labels_train.append(labels[i])
        
    seqs_test, labels_test = [], []
    for i in test_index:
        seqs_test.append(seqs[i])
        labels_test.append( labels[i])
    
    #y_pred_kf = np.zeros(shape=(len(labels_test), params['num_classes']))
    X_train = protseq_to_vec(seqs_train, padding_position="post", maxlen=params['maxlen'])
    X_test = protseq_to_vec(seqs_test, padding_position="post", maxlen=params['maxlen'])
    
    y_train = keras.utils.to_categorical(labels_train, params['num_classes'])
    y_test = keras.utils.to_categorical(labels_test, params['num_classes'])
    
    modelfile = './model/ec/ec_trainsformer_1024_{}.h5'.format("post")
    score = transformer_predictor(X_train, y_train, X_test, y_test, modelfile, params)
    pred = np.argmax(score, 1)
    #info = "\nEC v NOtEC dataset, kf-5: maxlen=1024, post padding\n"
    displayMetrics(np.argmax(y_test, 1), pred)
    #y_pred_kf += pred
   
    X_train, y_train = padding_seqs(seqs_train, labels_train, 
                                    padding_position="pre", 
                                    vocab_size=params['vocab_size'], 
                                    maxlen=params['maxlen'])
    X_test, y_test = padding_seqs(seqs_test, labels_test, 
                                    padding_position="pre", 
                                    vocab_size=params['vocab_size'], 
                                    maxlen=params['maxlen'])
    modelfile = './model/mlec/xiao_mlec_trainsformer_{}_{}.h5'.format(maxlen, "pre")
    score = transformer_predictor(X_train, y_train, X_test, y_test, modelfile, params)
    pred = (score>0.5).astype(float)
    info = "\nXiao's dataset, kf-{}: maxlen={}, pre padding\n".format(k, maxlen)
    displayMLMetrics(y_test, pred, "ml_result.txt", info)
    y_pred_kf += pred
    
#y_pred_kf /= 10
    #y_pred_kf = (y_pred_kf>0.5).astype(float)
    # print ensember performance
    #info = "\nXiao's dataset, kf-{}: ensember classifier performance\n".format(k, maxlen)
    #displayMLMetrics(y_test, y_pred_kf, "ml_result.txt", info)
    y_score = np.concatenate((y_score, score))
    y_true = np.concatenate((y_true, y_test))

#info = "\nXiao's dataset,Total predict performance\n"       
#displayMLMetrics(y_true, y_pred, "ml_result.txt", info)
displayMetrics(np.argmax(y_true,1), np.argmax(y_score, 1))
"""