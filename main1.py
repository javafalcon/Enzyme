#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:17:23 2020

@author: weizhong
"""

from Capsule import CapsuleLayer, squash, Length, Mask, margin_loss

from prepareDataset import load_Kf_data

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, backend
from tensorflow.keras.utils import to_categorical

import numpy as np

from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score,recall_score,precision_score

def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          name='primaryCap_conv2d')(inputs)
    dim = output.shape[1]*output.shape[2]*output.shape[3]
    outputs = layers.Reshape(target_shape=(dim//dim_vector,dim_vector), name='primaryCap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

def CapsNet(input_shape, num_classes, num_routing):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='valid',
                         activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=64, kernel_size=5, 
                            strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=num_classes, dim_vector=16, num_routing=num_routing,
                            name='digitcaps')(primarycaps)
    out_caps = Length(name='out_caps')(digitcaps)
    
    # Decoder network
    y = layers.Input(shape=(num_classes,))
    masked = Mask()([digitcaps, y])
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)
    
    return models.Model([x,y], [out_caps, x_recon])

def trainAndTest(model, data, lr, lam_recon, batch_size, epochs):
    (x_train, y_train),(x_test, y_test) = data
    
    model.compile(optimizer=optimizers.Adam(lr=lr),
                 loss=[margin_loss, 'mse'],
                 loss_weights=[1., lam_recon],
                 metrics={'out_caps': 'accuracy'})
    
    # callbacks
    #log = callbacks.CSVLogger('./result/PDNA-543/log.csv')
    #tb = callbacks.TensorBoard(log_dir='./result/PDNA-543/tensorboard-logs',
    #                           batch_size=batch_size, histogram_freq=1)
    #checkpoint = callbacks.ModelCheckpoint('./result/PDNA-543/weights-{epoch:02d}.h5',
    #                                       save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
    
    model.fit([x_train, y_train], [y_train, x_train],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[[x_test,y_test],[y_test,x_test]],
              callbacks=[lr_decay])
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=batch_size)
    
    return y_pred

def lr_schedule(epoch):
    lr = 1e-3
    return lr*0.9*epoch

def writeMetrics(metricsFile, y_true, y_pred, noteInfo=''):
    predicts = np.argmax(y_pred, 1)
    predicts = predicts[:,0]
    labels = np.argmax(y_true, 1)
    cm=confusion_matrix(labels,predicts)
    with open(metricsFile,'a') as fw:
        if noteInfo:
            fw.write(noteInfo + '\n')
            
        for i in range(7):
            for j in range(7):
                fw.write(str(cm[i,j]) + "\t" )
            fw.write("\n")
            
        fw.write("ACC: %f "%accuracy_score(labels,predicts))
        fw.write("\nRecall: %f "%recall_score(labels,predicts))
        fw.write("\nPre: %f "%precision_score(labels,predicts))
        

if __name__ == "__main__":
    row, col, channels = 21, 21, 1
    kfold = 5
    num_classes = 7
    metricsFile = 'result.txt'
    (X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf) = load_Kf_data(kfold=5, random_state=143)
    y_pred = np.zeros((0,7))
    y_true = np.zeros((0,7))
    
    for k in range(kfold):
        train_X = X_train_Kf[k].reshape((-1, row, col, channels))
        test_X = X_test_Kf[k].reshape((-1, row, col, channels))
        y_train = to_categorical(y_train_Kf[k], num_classes=num_classes)
        y_test = to_categorical(y_test_Kf[k], num_classes=num_classes)
        
        model = CapsNet(input_shape=[row,col,channels], num_classes=num_classes, num_routing=5)
        model.summary()
    
        pred = trainAndTest(model, ((train_X, y_train), (test_X, y_test)),
                     lr=0.001, lam_recon=0.35, batch_size=48, epochs=10)
        y_pred = np.concatenate((y_pred, pred))
        y_true = np.concatenate((y_true,y_test))
        
        noteInfo = "{}/{} cross training-testing:".format(k, kfold)
        writeMetrics(metricsFile, y_test, pred, noteInfo)
    
    noteInfo = "\nTotal validation result:"
    writeMetrics(metricsFile, y_true, y_pred, noteInfo)