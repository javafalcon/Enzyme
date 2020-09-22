# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 07:43:03 2020

@author: lwzjc
"""
import re
import numpy as np
from prepareDataset import readSLEnzyme, readMLEnzyme
from nlp_transformer import Encoder, create_padding_mask
from transformer import TokenAndPositionEmbedding, TransformerBlock
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras import layers

# Only consider the 21 amino acids:"ARNDCQEGHILKMFPSTWYVX" and [start],[end]
vocab_size = 24  

def load_data(n_split, test_size=0.2, random_state=None):
    """
    #nr40 = ['data/slec_{}_40.fasta'.format(i) for i in range(1,8)]
    nr40 = ['data/slec_{}_40.fasta'.format(i) for i in range(1,4)]
    nr60 = ['data/slec_{}_60.fasta'.format(i) for i in range(4,7)]
    nr80 = ['data/slec_{}_80.fasta'.format(i) for i in range(7,8)]
    files = nr40+nr60+nr80
    prot_seqs, prot_labels = readSLEnzyme(files)
    """
    prot_seqs, prot_labels = readMLEnzyme()
    amino_acids = "ARNDCQEGHILKMFPSTWYVX"
    
    regexp = re.compile('[^ARNDCQEGHILKMFPSTWYVX]')
    X, y = [], []
    for key in prot_seqs.keys():
        seq = prot_seqs[key]
        seq = regexp.sub('X', seq)
        t = [22]
        for a in seq:
            t.append(amino_acids.index(a) + 1)
        t.append(23)
        X.append(t)
        y.append(prot_labels[key])
    
    X_trains, X_tests = [], []
    y_trains, y_tests = [], []
    """
    sss = StratifiedShuffleSplit(n_splits=n_split, 
                                 test_size=test_size, 
                                 random_state=random_state)
    """
    sss = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in sss.split(X, y):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = [], []
        y_train, y_test = [], []
        for i in train_index:
            X_train.append(X[i])
            y_train.append(y[i])
        for j in test_index:
            X_test.append(X[j])
            y_test.append(y[j])
            
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
        
    return (X_trains, y_trains), (X_tests, y_tests)


# Download and prepare dataset
maxlen = 800  # Only consider the first 1000 amino acids of each protein sequence

#(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
(X_trains, y_trains), (X_tests, y_tests) = load_data(n_split=5, 
                                                     random_state=0)
x_train, y_train = X_trains[0], y_trains[0]
x_val, y_val = X_tests[0], y_tests[0]

#my_class_weight = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train).tolist()
#class_weight_dict = dict(zip([x for x in np.unique(y_train)], my_class_weight))


#y_train = keras.utils.to_categorical(y_train, 7)
#y_val = keras.utils.to_categorical(y_val, 7)
y_train = np.array(y_train)
y_val = np.array(y_val)

#sample_weight = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen,
                                                     padding="post",
                                                     truncating="post")
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen,
                                                   padding="post",
                                                   truncating="post")

# Create classifier model using transformer layer
embed_dim = 64  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 1028  # Hidden layer size in feed forward network inside transformer

keras.backend.clear_session()
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

model = keras.Model(inputs=inputs, outputs=outputs)
# Train and Evaluate
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    x_train, y_train, batch_size=32, epochs=10, 
    validation_data=(x_val, y_val)
)