# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:20:26 2020

@author: Administrator
"""

from SeqFormulate import DAA_chaosGraph
import numpy as np
from prepareData import readMLEnzyme

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import KFold
from sklearn.metrics import hamming_loss, accuracy_score
def load_mlec():
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

def resnet_layer(inputs, num_filters, kernel_size=3, strides=1,
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

def resnet_v1(input_shape, depth, num_classes=2):
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
    num_filters = 32
    num_res_blocks = int((depth-2)/6)
    
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet_layer(inputs, num_filters)
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
    #x = layers.AveragePooling2D()(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid',
                           kernel_initializer='he_normal')(y)
    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    return model

x, y = load_mlec()
lr = 0.001
k = 1
y_pred = np.zeros((0, 7))
y_true = np.zeros((0, 7))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(x,y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    tf.keras.backend.clear_session()
    model = resnet_v1(input_shape=(21, 21, 1), depth=20, num_classes=2)
    model.summary()
    modelfile = './model/mlec/weights-mlec-{}.h5'.format(k)
    model.compile(optimizer=Adam(learning_rate=lr),
         loss='categorical_crossentropy',
         metrics=['accuracy'])

    lr_decay = LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))

    checkpoint = ModelCheckpoint(modelfile, monitor='val_loss',
                                   save_best_only=True, 
                                   save_weights_only=True, 
                                   verbose=1)
    model.fit(x_train, y_train,
              batch_size=50,
              epochs=10,
              validation_data=[x_test, y_test],
              callbacks=[checkpoint, lr_decay])
    
    pred = model.predict(x_test)
    
    y_pred = np.concatenate((y_pred, pred))
    y_true = np.concatenate((y_true,y_test))
    
with open('ml_result.txt', 'a') as fw:
    fw.write("hamming loass = {}\n".format(hamming_loss(y_true, y_pred)))
    fw.write("subset accuracy = {}\n".format( accuracy_score(y_true, y_pred)))
    
