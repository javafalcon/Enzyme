# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:46:24 2020

@author: lwzjc
"""
from prepareDataset import readSLEnzyme, load_Kf_data, load_MISL
from SeqFormulate import DAA_chaosGraph
from over_sampling import singleLabelSMOTE
import re
import numpy as np

from sklearn.utils import shuffle
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

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

def loadSL():
    nr40 = ['data/slec_{}_40.fasta'.format(i) for i in range(1,8)]
    prot_seqs, prot_labels = readSLEnzyme(nr40)
    seqs, labels = [], []
    for key in prot_seqs.keys():
        seqs.append(prot_seqs[key])
        labels.append(prot_labels[key])
    x = np.ndarray(shape=(len(seqs), 21, 21, 2))  
        
    x[:,:,:,0] = DAA_chaosGraph(seqs)
    x[:,:,:,1] = daa(seqs)
            
    y = np.array(labels)
    
    return x, y

def underSample(X, rate):
    N = X.shape[0]
    n = int(np.round(N*rate))
    X = shuffle(X)
    rng = np.random.default_rng()
    return rng.choice(X, n, replace=False)
      
def resample(x, y, k_neighbors, rate):
    """
    numbers of 7 classes: [ 3995., 10219.,  8477.,  1559.,  1177.,  1674.,   690.]
    under-sample for NO.0,1,2
    over-sample for NO.6
    
    Parameters
    ----------
    x : ndarry, shape (n_samples, n_features)
    y : ndarry, shape (n_samples, )

    Returns
    -------
    None.

    """
    # 0-th class
    indx = (y==0)
    x_0 = underSample(x[indx], 0.35)
    y_0 = np.zeros((len(x_0),))
    
    # 1-th class
    indx = (y==1)
    x_1 = underSample(x[indx], 0.13)
    y_1 = np.ones((len(x_1),))
    
    # 2-th class
    indx = (y==2)
    x_2 = underSample(x[indx], 0.16)
    y_2 = np.ones((len(x_2),)) * 2
    
    # 3-th class
    indx = (y==3)
    x_3 = underSample(x[indx], 0.9)
    y_3 = np.ones((len(x_3),)) * 3
    
    # 4-th class
    indx = (y==4)
    x_4 = x[indx]
    y_4 = np.ones((len(x_4),)) * 4
    
    # 5-th class
    indx = (y==5)
    x_5 = underSample(x[indx], 0.82)
    y_5 = np.ones((len(x_5),)) * 5
    
    # 6-th class
    indx = (y==6)
    x_6 = singleLabelSMOTE(x[indx], k_neighbors=k_neighbors, rate=rate)
    y_6 = np.ones((len(x_6),)) * 6
    
    x_new = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6))
    y_new = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6))
    
    return x_new ,y_new

def over_resample(x, y, k_neighbors):
    """
    numbers of 7 classes: [ 3995., 10219.,  8477.,  1559.,  1177.,  1674.,   690.]
    under-sample for NO.0,1,2
    over-sample for NO.6
    
    Parameters
    ----------
    x : ndarry, shape (n_samples, n_features)
    y : ndarry, shape (n_samples, )

    Returns
    -------
    None.

    """
    # 0-th class
    indx = (y==0)
    x_0 = singleLabelSMOTE(x[indx], k_neighbors=k_neighbors, rate=1.5)
    y_0 = np.zeros((len(x_0),))
    print("genrate {} {}-th samples".format(len(x_0), 0))
    # 1-th class
    indx = (y==1)
    x_1 =x[indx]
    y_1 = np.ones((len(x_1),))
    print("genrate {} {}-th samples".format(len(x_1), 1))
    # 2-th class
    indx = (y==2)
    x_2 = singleLabelSMOTE(x[indx], k_neighbors=k_neighbors, rate=0.2)
    y_2 = np.ones((len(x_2),)) * 2
    print("genrate {} {}-th samples".format(len(x_2), 2))
    # 3-th class
    indx = (y==3)
    x_3 = singleLabelSMOTE(x[indx], k_neighbors=k_neighbors, rate=5.5)
    y_3 = np.ones((len(x_3),)) * 3
    print("genrate {} {}-th samples".format(len(x_3), 3))
    # 4-th class
    indx = (y==4)
    x_4 = singleLabelSMOTE(x[indx], k_neighbors=k_neighbors, rate=7.7)
    y_4 = np.ones((len(x_4),)) * 4
    print("genrate {} {}-th samples".format(len(x_4), 4))
    # 5-th class
    indx = (y==5)
    x_5 = singleLabelSMOTE(x[indx], k_neighbors=k_neighbors, rate=5)
    y_5 = np.ones((len(x_5),)) * 5
    print("genrate {} {}-th samples".format(len(x_5), 5))
    # 6-th class
    indx = (y==6)
    x_6 = singleLabelSMOTE(x[indx], k_neighbors=k_neighbors, rate=14)
    y_6 = np.ones((len(x_6),)) * 6
    print("genrate {} {}-th samples".format(len(x_6), 6))
    x_new = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6))
    y_new = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6))
    
    return x_new ,y_new

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
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(y)
    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def predict(x_train, y_train, x_test, y_test, params):
    tf.keras.backend.clear_session()
    resnetparams = {
        'num_filters': params.get('num_filters', 32), 
        'num_classes': params.get('num_classes', 2), 
        'dropout': params.get('dropout', 0.25),
        'kernel_size': params.get('kernel_size', 3)
        }
    model = resnet_v1(input_shape=(10,21, 21), depth=20, netparam=resnetparams)
    model.summary()
    modelfile = params['modelfile'] 
    lr = params['lr']   
               
    model.compile(optimizer=Adam(learning_rate=lr),
         loss= 'categorical_crossentropy',
         metrics=['accuracy'])
            
    lr_decay = LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))

    checkpoint = ModelCheckpoint(modelfile, monitor='val_loss',
                                   save_best_only=True, 
                                   save_weights_only=True, 
                                   verbose=1)
    
    x_train, y_train = shuffle(x_train, y_train)
    
    model.fit(x_train, y_train,
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              validation_data=[x_test, y_test],
              #validation_split=0.1,
              callbacks=[checkpoint, lr_decay])
    
    model.load_weights(modelfile)
    pred = model.predict(x_test)
    return pred

def resnetWithAttention_main(x, y, params, kfold=3 ):
    lr = 0.001
    k = 0
    num_classes = params.get('num_classes',2)
    y_pred = np.zeros((0, num_classes))
    y_true = np.zeros((0, num_classes))
    
    (X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf) = load_Kf_data(x, y, kfold=kfold, random_state=42)
    
    for k in range(kfold):
        x_train, x_test = X_train_Kf[k], X_test_Kf[k]
        y_train, y_test = y_train_Kf[k], y_test_Kf[k]
        
        #x_train = x_train.reshape((-1, 21*21*2))
        y_test = to_categorical(y_test, 7)
        y_train = to_categorical(y_train, 7)
        pred = np.zeros(y_test.shape)
        
        for j in range(1):
            #x_train_res, y_train_res = over_resample(x_train, y_train, 
            #                                    k_neighbors=params['k_neighbors']
            #                                    )
            #x_train_res = x_train_res.reshape((-1, 21, 21, 2))
            #y_train_res = to_categorical(y_train_res, 7)
                        
            nnparams = {
                'modelfile': './model/slec/{}-{}.h5'.format(k,j),
                'lr': lr,
                'batch_size': params['batch_size'],
                'epochs': params['epochs'],
                'num_filters': params['num_filters'], 
                'num_classes': params['num_classes'], 
                'dropout': params['dropout'],
                'kernel_size': params['kernel_size']
                }
            
            predscore = predict(x_train, y_train, x_test, y_test, nnparams)
            pred += (predscore > 0.5).astype(float)
            
        y_pred = np.concatenate((y_pred, pred))
        y_true = np.concatenate((y_true,y_test))
             
    y_t = np.argmax(y_true,axis=1)
    y_p = np.argmax(y_pred,axis=1)
    
    info = "----- classify single label for Enzyme by resampling -------"
    for key in params.keys():
        info += key + ": " + str(params.get(key, None)) + '\n'
      
    with open('sl_result.txt', 'a') as fw:
        fw.write(info)
        cm=metrics.confusion_matrix(y_t, y_p)
        for i in range(num_classes):
                for j in range(num_classes):
                    fw.write(str(cm[i,j]) + "\t" )
                fw.write("\n")
            
        fw.write("ACC = {} \n".format(metrics.accuracy_score(y_t,y_p)))

    return (y_t, y_p)

if __name__ == "__main__":
    nr40 = ['data/slec_{}_40.fasta'.format(i) for i in range(1,8)]
    x,y=load_MISL(nr40)
    params = {'k_neighbors': 5, 'rate': 1, 
              'batch_size': 100, 'epochs': 20,
              'kernel_size': 5, 'num_filters': 64, 
              'num_classes': 7, 'dropout': 0.25,
              } 
    accuracy = resnetWithAttention_main(x, y, params=params)
    #x_over, y_over = over_resample(x.reshape(-1,21*21*2),y,5)
