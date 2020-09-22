# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 07:06:03 2020

@author: lwzjc
"""

from resnet import resnet_layer
import tensorflow as tf
from tensorflow.keras import layers, models
from prepareDataset import readMLEnzyme, readSLEnzyme
import numpy as np
import random
from prepareDataset import gen_bag_inst, load_Kf_data
from tensorflow.keras.utils import to_categorical


files = ['data/slec_{}_40.fasta'.format(i) for i in range(1,8)]    
X,Y = load_MISL(files)

(X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf) = load_Kf_data(X, Y)    
num_filters = 32
num_res_blocks = 3
num_classes = 2
epochs = 20
batch_size = 32
num_samples = X_train.shape[0]

#x_input = tf.placeholder(tf.float32, (None, 1, 21, 21))
#target = tf.placeholder(tf.float32, (None, num_classes))

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

X_train, X_test = X_train_Kf[0], X_test_Kf[0]
y_train, y_test = y_train_Kf[0], y_test_Kf[0]
y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)

total_batch = np.ceil(num_samples/batch_size)
for _ in range(epochs):
        for i in range(total_batch):
            start = i*batch_size
            end = (i+1)*batch_size if (i+1)*batch_size < num_samples else num_samples
            batch_x, batch_y =  X_train[start:end], y_train[start:end]
            
            for bx in batch_x:
                for i in range(10):
                    prediction = sess.run(y_predict, fetches={x_input:bx[i,:,:]})
                
    

