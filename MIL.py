# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 07:06:03 2020

@author: lwzjc
"""

from resnet import resnet_layer
import tensorflow as tf
from tensorflow.keras import layers, models
from prepareDataset import readMLEnzyme
from SeqFormulate import seqAAOneHot

def gen_bag_inst(seq, k, num_inst):
    seqs = []
    seqs.append(seq[:k])
    seqs.append(seq[-k:])
    for _ in range(num_inst-2):
        
    
def load_mlec_nr(nr=80):
    mlec_seqs, mlec_labels = readMLEnzyme()
    seqs, labels = [], []
    
    for seq_record in SeqIO.parse('data/mlec_{}.fasta'.format(nr), 'fasta'):
        s = seq_record.id
        pid = s.split(' ')
        protId = pid[0]
        seqs.append(mlec_seqs[protId])
        labels.append(mlec_labels[protId])
    
    x = np.ndarray(shape=(len(seqs), 32, 20, 15))    
    x[:,:,:,0] = DAA_chaosGraph(seqs)
    
    y = np.array(labels)
    return x, y

num_filters = 32
num_res_blocks = 3
num_classes = 2
epochs = 20

x_input = tf.placeholder(tf.float32, (None, 32, 20))
target = tf.placeholder(tf.float32, (None, num_classes))

x = resnet_layer(x_input, num_filters)
# Instantiate teh stack of residual units
for stack in range(3):
    for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0: # first layer but not first stack
            strides = 2 # downsample
        y = resnet_layer(x, num_filters, strides=strides)  
        y = resnet_layer(y, num_filters, activation=None)
        
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
x = layers.GlobalAveragePooling2D()(x)

y_predict = layers.Dense(num_classes, activation='softmax',
                       kernel_initializer='he_normal')(x)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for _ in range(epochs):
        batch_x, batch_y =     
    

