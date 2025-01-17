# -*- coding: utf-8 -*-from tensorflow.keras.utils import to_categorical
"""
Created on Fri Jun 19 09:12:35 2020

@author: lwzjc
"""

from SeqFormulate import DAA_chaosGraph
import numpy as np
import re
from prepareDataset import readSLEnzyme, load_Kf_data

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection, RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score,recall_score,precision_score
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
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

def resnet_v1(input_shape, depth, num_classes=1, num_filters=32, kernel_size=3):
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
    num_res_blocks = int((depth-2)/6)
    
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet_layer(inputs, num_filters, kernel_size=kernel_size)
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
    
    ax = layers.Flatten()(ax)
    ax = layers.Dropout(0.1)(ax)
    outputs = layers.Dense(num_classes, activation='softmax')(ax)
    
    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=10, pool_size=8):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = tf.keras.Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(2):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #x = layers.AveragePooling2D(pool_size=pool_size)(x)
    ax = layers.GlobalAveragePooling2D()(x)
    
    ax = layers.Dense(num_filters_in//4, activation='relu')(ax)
    ax = layers.Dense(num_filters_in, activation='softmax')(ax)
    ax = layers.Reshape((1,1,num_filters_in))(ax)
    ax = layers.Multiply()([ax, x])
    y = layers.Flatten()(ax)

    outputs = layers.Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model 
    
nr40 = ['data/slec_{}_40.fasta'.format(i) for i in range(1,8)]
#nr60 = ['data/slec_{}_60.fasta'.format(i) for i in range(4,8)]
prot_seqs, prot_labels = readSLEnzyme(nr40)
seqs, labels = [], []
for key in prot_seqs.keys():
    seqs.append(prot_seqs[key])
    labels.append(prot_labels[key])
x = np.ndarray(shape=(len(seqs), 21, 42, 1))  
    
x[:,:,:21,0] = DAA_chaosGraph(seqs)
x[:,:,21:,0] = daa(seqs)
        
y = np.array(labels)
(X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf) = load_Kf_data(x, y, random_state=42)
lr = 0.001
k = 0
num_classes = 7
y_pred = np.zeros((0, num_classes))
y_true = np.zeros((0, num_classes))

#sm = SMOTE(sampling_strategy='not majority')
#oss = OneSidedSelection('majority', random_state=42)
#rus = RandomUnderSampler(random_state=42)
for k in range(1):
    x_train, x_test = X_train_Kf[k], X_test_Kf[k]
    y_train, y_test = y_train_Kf[k], y_test_Kf[k]
    
    #x_train = x_train.reshape((-1, 21*21*2))
    #x_res, y_res = oss.fit_resample(x_train, y_train)
    #x_res, y_res = rus.fit_resample(x_train, y_train)
    #x_res, y_res = sm.fit_resample(x_res, y_res)
    #x_train = x_res.reshape((-1, 21,21,2))
    #y_train = to_categorical(y_res, num_classes)
    
    my_class_weight = compute_class_weight('balanced',np.unique(y_train),y_train).tolist()
    class_weight_dict = dict(zip([x for x in np.unique(y_train)], my_class_weight))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
        
    tf.keras.backend.clear_session()
    model = resnet_v1(input_shape=(21, 42, 1), depth=20, 
                      num_classes=num_classes,
                      num_filters=64,
                      kernel_size=5)
    model.summary()
    modelfile = './model/slec/weights-slec-{}.h5'.format(k)
   
    model.compile(optimizer=Adam(learning_rate=lr),
         loss='categorical_crossentropy',
         metrics=['accuracy'])

    lr_decay = LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))

    checkpoint = ModelCheckpoint(modelfile, monitor='val_loss',
                                   save_best_only=True, 
                                   save_weights_only=True, 
                                   verbose=1)
    
    x_train, y_train = shuffle(x_train, y_train)
    history = model.fit(x_train, y_train,
              batch_size=100,
              epochs=20,
              validation_data=[x_test, y_test],
              class_weight=class_weight_dict,
              callbacks=[checkpoint])
    
    #model.load_weights(modelfile)
    pred = model.predict(x_test)
    k += 1
    y_pred = np.concatenate((y_pred, pred))
    y_true = np.concatenate((y_true, y_test))

from tools import plot_history
plot_history(history)

with open('sl_result.txt', 'a') as fw:
    y_t = np.argmax(y_true,axis=1)
    y_p = np.argmax(y_pred,axis=1)
    #y_p = (y_pred > 0.5).astype(int)
    #y_t = y_true
    cm=confusion_matrix(y_t, y_p)
    for i in range(num_classes):
            for j in range(num_classes):
                fw.write(str(cm[i,j]) + "\t" )
            fw.write("\n")
            
    fw.write("ACC = {} \n".format(accuracy_score(y_t,y_p)))
    '''fw.write("micro recall = {}\n".format(recall_score(y_t, y_p, average='micro')))
    fw.write("macro recall = {}\n".format(recall_score(y_t, y_p, average='macro')))
    fw.write("micro precision = {}\n".format(precision_score(y_t, y_p, average='micro')))
    fw.write("micro precision = {}\n".format(precision_score(y_t, y_p, average='macro')))
    fw.write("micro AUC = {}\n".format(roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')))
    fw.write("macro AUC = {}\n".format(roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')))
    '''
