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

def load_mlec_4479(firstly_load=False):
    if firstly_load:
        matdata=sio.loadmat('data/4479_label.mat')
        labels = matdata['label']
        seqs = []
        for record in SeqIO.parse('data/4479(0.9).fasta','fasta'):
            seqs.append(str(record.seq))
        
        x = np.ndarray(shape=(len(seqs), 21, 21, 2))    
        x[:,:,:,0] = DAA_chaosGraph(seqs)
        x[:,:,:,1] = daa(seqs)
        y = np.array(labels, dtype=float)
        
        np.savez('mlec_4479.npz', x=x, y=y)
    else:
        data = np.load('mlec_4479.npz')
        x, y = data['x'], data['y']
    return x,y
    
def load_mlec_nr(nrfile='data/melc_90.fasta', npzfile='melc_nr90.npz', description=False, firstly_load=False):
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
        
        x = np.ndarray(shape=(len(seqs), 21, 21, 2))    
        x[:,:,:,0] = DAA_chaosGraph(seqs)
        x[:,:,:,1] = daa(seqs)
        y = np.array(labels)
        np.savez(nzfile, x=x, y=y)
    else:
        data = np.load(npzfile)
        x, y = data[x], data[y]
    return x, y
        
def load_mled_v2():
    mlec_seqs, mlec_labels = readMLEnzyme()
    seqs, labels = [], []
    for key in mlec_seqs.keys():
        seqs.append(mlec_seqs[key])
        labels.append(mlec_labels[key])
    x = DAA_chaosGraph(seqs)
    y = np.array(labels)
    return x, y

def lr_schedule(epoch):
    lr = 1e-3
    return lr*0.9*epoch

def resnet_layer(inputs, num_filters, kernel_size=5, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
    ''' 2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    '''
    conv = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                         padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
        
    return x

def resnet_v1(input_shape, depth, netparam={}):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth-2)%6 != 0:
        raise ValueError('depth should be 6n+2')
    # Start model definition.
    num_filters=netparam.get('num_filters', 32)
    kernel_size=netparam.get('kernel_size', 3)
    num_classes=netparam.get('num_classes', 2)
    dropout = netparam.get('dropout', None)
    num_res_blocks = int((depth-2)/6)
    
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet_layer(inputs, num_filters)
    # Instantiate teh stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0: # first layer but not first stack
                strides = 2 # downsample
            y = resnet_layer(x, num_filters, kernel_size=kernel_size, strides=strides)  
            y = resnet_layer(y, num_filters, kernel_size=kernel_size, activation=None)
            
            if stack > 0 and res_block == 0: # first layer but not first stack
                # linear projection residual shortcut connection to match
                # change dims
                x = resnet_layer(x, num_filters, kernel_size=1, strides=strides,
                                 activation=None, batch_normalization=False)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)           
        num_filters *= 2
        
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    ax = layers.GlobalAveragePooling2D()(x)
    #ax = layers.GlobalMaxPool2D()(x)
    
    ax = layers.Dense(num_filters//8, activation='relu')(ax)
    ax = layers.Dense(num_filters//2, activation='softmax')(ax)
    ax = layers.Reshape((1,1,num_filters//2))(ax)
    ax = layers.Multiply()([ax, x])
    y = layers.Flatten()(ax)
    if dropout:
        y = layers.Dropout(dropout)(y)
    outputs = layers.Dense(num_classes, activation='sigmoid',
                           kernel_initializer='he_normal')(y)
    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def myloss(weight):
    def weightloss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()  
        return -tf.math.reduce_mean( (y_true*tf.math.log(epsilon + y_pred) + (1-y_true)*tf.math.log(1-y_pred+epsilon)) * weight)
    return weightloss

def resnetWithAttention_main(x, y, using_weight=False, params=None):
    # 4478 enzyme sequences with 7 labels
    # [1076, 2814, 1924,  854,  237,  205,   49]
    # [2.6, 1, 1.5, 3.3, 12, 14, 57]
    lr = 0.001
    k = 0
    
    y_pred = np.zeros((0, params.get('num_classes',2)))
    y_true = np.zeros((0, params.get('num_classes',2)))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        tf.keras.backend.clear_session()
        model = resnet_v1(input_shape=(21, 21, 2), depth=20, netparam=params)
        model.summary()
        modelfile = './model/mlec/weights-4479-{}.h5'.format(k)
               
        if using_weight:
            ntarg = np.sum(y_train, axis=0)
            weight = np.floor(np.max(ntarg)/ntarg)

            model.compile(optimizer=Adam(learning_rate=lr),
                 loss= myloss(weight),
                 metrics=['accuracy'])
            c=0.5
            info = '\n--- use 4479 dataset by using weight binary crossentropy---\n'
        else:
            model.compile(optimizer=Adam(learning_rate=lr),
                 loss= 'binary_crossentropy',
                 metrics=['accuracy'])
            c = np.sum(y_true,axis=0)/(len(y_true))
            info = '\n--- use 4479 dataset by binary corssentropy---\n'
            
        lr_decay = LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
    
        checkpoint = ModelCheckpoint(modelfile, monitor='val_loss',
                                       save_best_only=True, 
                                       save_weights_only=True, 
                                       verbose=1)
        
        x_train, y_train = shuffle(x_train, y_train)
        
        model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=20,
                  validation_data=[x_test, y_test],
                  callbacks=[checkpoint, lr_decay])
        
        model.load_weights(modelfile)
        pred = model.predict(x_test)
        k += 1
        y_pred = np.concatenate((y_pred, pred))
        y_true = np.concatenate((y_true,y_test))
             
    y_p = (y_pred>c).astype(float)
    
    for key in params.keys():
        info += key + ": " + str(params.get(key, None)) + '\n'
      
    with open('ml_result.txt', 'a') as fw:
        fw.write(info)
        fw.write("hamming loss = {}\n".format(metrics.hamming_loss(y_true, y_p)))
        fw.write("subset accuracy = {}\n".format( metrics.accuracy_score(y_true, y_p)))
        fw.write("macro average precision_score: {}\n".format(metrics.average_precision_score(y_true,y_p,average="macro")))
        fw.write("micro average precisioin_score: {}\n".format(metrics.average_precision_score(y_true,y_p,average="micro")))
    
    return metrics.accuracy_score(y_true, y_p)
def statInfo():
    #statlen = np.zeros((7,))
    statlen = np.zeros((2,))
    for seq_record in SeqIO.parse('data/mlec.fasta', 'fasta'):
        seq = str(seq_record.seq)
        '''if len(seq) < 1000:
            statlen[0] += 1
        elif len(seq) < 2000:
            statlen[1] += 1
        elif len(seq) < 3000:
            statlen[2] += 1
        elif len(seq) < 4000:
            statlen[3] += 1
        elif len(seq) < 5000:
            statlen[4] += 1
        elif len(seq) < 6000:
            statlen[5] += 1
        else:
            statlen[6] += 1'''
        if len(seq) < 4225:
            statlen[0] += 1
        else:
            statlen[1] += 1
    print(statlen)
    return statlen
if __name__ == "__main__":    
    #weight = np.array([1, 1, 1, 1, 5, 15])
    #weight = np.array([2, 1, 1, 3, 12, 14, 57])
    x, y = load_mlec_4479(False) 
    x, y = shuffle(x, y)
    best_subacc = 0.
    best_param = {'kernel_size': 0, 'num_filters': 0,  'dropout': 0} 
    for k in [3, 5, 7]:
        for f in [16, 32, 64]:
            for d in [0.25, 0.45, 0.65]:
                params = {'kernel_size': k, 'num_filters': f, 'num_classes': 7, 'dropout': d} 
                subacc = resnetWithAttention_main(x, y, params=params, using_weight=True)
                if subacc > best_subacc:
                    best_subacc = subacc
                    best_param['kernel_size'] = k
                    best_param['num_filters'] = f
                    best_param['dropout'] = d
    
    # report the best configuration
    print("Best: %f using %s" % (best_subacc, best_param))
    