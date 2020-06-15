#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:17:23 2020

@author: weizhong
"""

from Capsule import CapsuleLayer, squash, Length, Mask, margin_loss

from prepareDataset import load_Kf_data, load_data, load_SL_EC_data, load_ML_SL_EC_data
from resnet import resnet_v1

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, backend
from tensorflow.keras.utils import to_categorical

import numpy as np

from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score,recall_score,precision_score
from sklearn.model_selection import train_test_split

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

def CapsTrainAndTest(model, data, modelfile, lr, lam_recon, batch_size, epochs):
    (x_train, y_train),(x_test, y_test) = data
    
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                 loss=[margin_loss, 'mse'],
                 loss_weights=[1., lam_recon],
                 metrics={'out_caps': 'accuracy'})
    
    # callbacks
    #log = callbacks.CSVLogger('./result/PDNA-543/log.csv')
    #tb = callbacks.TensorBoard(log_dir='./result/PDNA-543/tensorboard-logs',
    #                           batch_size=batch_size, histogram_freq=1)
    checkpoint = callbacks.ModelCheckpoint(modelfile,
                                           monitor='val_loss',
                                           save_best_only=True, 
                                           save_weights_only=True, 
                                           verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
    
    model.fit([x_train, y_train], [y_train, x_train],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[[x_test,y_test],[y_test,x_test]],
              callbacks=[checkpoint, lr_decay])
    
    model.load_weights(modelfile)
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=batch_size)
    
    return y_pred


def TrainAndTest(model, data, modelfile, class_weight, lr, batch_size, epochs):
    (x_train, y_train),(x_test, y_test) = data
    
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # callbacks
    #log = callbacks.CSVLogger('./result/PDNA-543/log.csv')
    #tb = callbacks.TensorBoard(log_dir='./result/PDNA-543/tensorboard-logs',
    #                           batch_size=batch_size, histogram_freq=1)
    checkpoint = callbacks.ModelCheckpoint(modelfile, monitor='val_loss',
                                           save_best_only=True, 
                                           save_weights_only=True, 
                                           verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_test,y_test],
              class_weight=class_weight,
              callbacks=[checkpoint, lr_decay])
    
    model.load_weights(modelfile)
    
    y_pred = model.predict(x_test, batch_size=batch_size)
    
    return y_pred

def lr_schedule(epoch):
    lr = 1e-3
    return lr*0.9*epoch

def writeMetrics(metricsFile, y_true, y_pred, noteInfo=''):
    predicts = np.argmax(y_pred, 1)
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
        
def CNN(input_shape, num_classes):
    x_input = tf.keras.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3,3), strides=1,
                          padding='same', activation='relu',
                          name='conv1')(x_input)
    x = layers.Conv2D(64, (3,3), strides=1,
                          padding='same', activation='relu',
                          name='conv2')(x)
    x  = layers.Conv2D(128, (3,3), strides=1,
                          padding='same', activation='relu',
                          name='conv3')(x)
    
    x = layers.MaxPool2D(pool_size=(2,2), strides=2)(x)
    
    x = layers.Conv2D(128, (3,3), strides=1,
                          padding='same', activation='relu',
                          name='conv4')(x)    
    x = layers.Conv2D(256, (3,3), strides=1,
                          padding='same', activation='relu',
                          name='conv5')(x)
    x = layers.Conv2D(256, (3,3), strides=1,
                          padding='same', activation='relu',
                          name='conv6')(x)
    
    x = layers.MaxPool2D(pool_size=(2,2), strides=2)(x)
    
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(512, activation='relu', name='dense1')(x)
    
    x = layers.Dense(1024, activation='relu', name='dense2')(x)
    
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(x_input, out)

def CnnNet(input_shape, n_class):
    regular = tf.keras.regularizers.l1(0.01)
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (5,5), strides=1,
                          padding='same', activation='relu', 
                          kernel_regularizer=regular,
                          name='conv1')(x)
    conv2 = layers.Conv2D(32, (5,5), padding='same', activation='relu', name='conv2')(conv1)
    pool1 = layers.MaxPool2D(pool_size=(2,2))(conv2)
    drop1 = layers.Dropout(0.25)(pool1)
    
    conv3 = layers.Conv2D(64,(5,5), padding='same',
                          activation='relu', name='conv3')(drop1)
    conv4 = layers.Conv2D(128, (5,5), activation='relu', name='conv4')(conv3)
    pool2 = layers.MaxPool2D()(conv4)
    drop2 = layers.Dropout(0.25)(pool2)
    
    flat = layers.Flatten()(drop2)
    dens1 = layers.Dense(512, activation='relu')(flat)
    drop3 = layers.Dropout(0.5)(dens1)
    out = layers.Dense(n_class, activation='softmax')(drop3)
    
    return models.Model(x, out)   

def classify_slec(random_state=143):
    row, col, channels = 21, 21, 1
    kfold = 5
    num_classes = 7
    metricsFile = 'result.txt'
    files=['data/slec_{}_40.fasta'.format(i) for i in range(1,8)]    
    x,y = load_SL_EC_data(files)
    (X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf) = load_Kf_data(x, y, kfold=5, random_state=random_state)
    y_pred = np.zeros((0, num_classes))
    y_true = np.zeros((0, num_classes))
    
    class_weight = [3,1,1,2,3,1.5,4.5]
    
    for k in range(kfold):
        train_X = X_train_Kf[k].reshape((-1, row, col, channels))
        test_X = X_test_Kf[k].reshape((-1, row, col, channels))
        y_train = to_categorical(y_train_Kf[k], num_classes=num_classes)
        y_test = to_categorical(y_test_Kf[k], num_classes=num_classes)
        
        tf.keras.backend.clear_session()
        #model = CapsNet(input_shape=[row,col,channels], num_classes=num_classes, num_routing=5)
        model = resnet_v1(input_shape=(row, col, channels), depth=20, num_classes=num_classes)
        model.summary()
        modelfile = './model/slec/weights-slec-{}.h5'.format(k)
        pred = TrainAndTest(model, ((train_X, y_train), (test_X, y_test)),
                            modelfile, class_weight,
                            lr=0.001, batch_size=50, epochs=10)
        y_pred = np.concatenate((y_pred, pred))
        y_true = np.concatenate((y_true,y_test))
        
        noteInfo = "{}/{} cross-validate predicting EC singal label:".format(k, kfold)
        writeMetrics(metricsFile, y_test, pred, noteInfo)
    
    noteInfo = "\nTotal validation result:"
    writeMetrics(metricsFile, y_true, y_pred, noteInfo)    
    
def classify_slec_bi(lr=0.001, random_state=143):
    row, col, channels = 21, 21, 1
    kfold = 5
    num_classes = 7
    metricsFile = 'result.txt'
    files1=['data/slec_{}_40.fasta'.format(i) for i in range(1,4)]
    files2 = ['data/slec_{}_60.fasta'.format(i) for i in range(4,8)]
    files = files1 + files2
    
    x,y = load_SL_EC_data(files)
    (X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf) = load_Kf_data(x, y, kfold=5, random_state=random_state)
    y_pred = np.zeros((0, num_classes))
    y_true = np.zeros((0, num_classes))
       
    for k in range(kfold):
        train_X = X_train_Kf[k].reshape((-1, row, col, channels))
        test_X = X_test_Kf[k].reshape((-1, row, col, channels))
        y_train = to_categorical(y_train_Kf[k], num_classes=num_classes)
        y_test = to_categorical(y_test_Kf[k], num_classes=num_classes)
        y_p = np.zeros(shape=y_test.shape)       
        for j in range(num_classes):
            train_y = to_categorical(y_train[:,j], 2)
            test_y = to_categorical(y_test[:,j], 2)
            
            tf.keras.backend.clear_session()
            #model = CapsNet(input_shape=[row,col,channels], num_classes=num_classes, num_routing=5)
            model = resnet_v1(input_shape=(row, col, channels), depth=20, num_classes=2)
            model.summary()
            modelfile = './model/slec/weights-slec-{}_{}.h5'.format(k,j)
            model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
            lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
    
            checkpoint = callbacks.ModelCheckpoint(modelfile, monitor='val_loss',
                                           save_best_only=True, 
                                           save_weights_only=True, 
                                           verbose=1)
            model.fit(train_X, train_y,
                      batch_size=50,
                      epochs=10,
                      validation_data=[test_X, test_y],
                      callbacks=[checkpoint, lr_decay])
    
            model.load_weights(modelfile)
            pred = model.predict(test_X, batch_size=50)
            y_p[:, j] = pred[:,1]
            
        y_pred = np.concatenate((y_pred, y_p))
        y_true = np.concatenate((y_true,y_test))
        
        noteInfo = "{}/{} cross-validate predicting EC singal label:".format(k, kfold)
        writeMetrics(metricsFile, y_test, y_p, noteInfo)
    
    noteInfo = "\nTotal validation result:"
    writeMetrics(metricsFile, y_true, y_pred, noteInfo)     
    
    return y_pred

def classify_ec(lr=0.001):
    row, col, channels = 21, 21, 1
    num_classes = 2
    metricsFile = 'result.txt'
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=59)
    
    x_train = x_train.reshape((-1, row, col, channels))
    y_train = to_categorical(y_train, num_classes=2)
    
    x_test = x_test.reshape((-1, row, col, channels))
    
    model = resnet_v1(input_shape=(row, col, channels), depth=20, num_classes=num_classes)
    model.summary()
    
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
    modelfile = './model/ec/weights-ec.h5'
    checkpoint = callbacks.ModelCheckpoint(modelfile, monitor='val_loss',
                                           save_best_only=True, 
                                           save_weights_only=True, 
                                           verbose=1)
    model.fit(x_train, y_train,
              batch_size=50,
              epochs=10,
              validation_split=0.1,
              callbacks=[checkpoint, lr_decay])
    
    model.load_weights(modelfile)
    pred = model.predict(x_test, batch_size=50)
    
    noteInfo = "\npredict EC and Not EC:"
    #y_true = np.argmax(y_test, 1)
    y_pred = np.argmax(pred, 1)
    cm = confusion_matrix(y_test, y_pred)
    with open(metricsFile, 'a') as fw:
        fw.write(noteInfo + "\n")
        for i in range(2):
            fw.write(str(cm[i,0]) + '\t' + str(cm[i,1]) + '\n')
        fw.write("ACC:%f\n"%accuracy_score(y_test, y_pred))
        fw.write("MCC:%f\n"%matthews_corrcoef(y_test, y_pred))
        
def classify_ML_SL_ec(lr=0.001):
    row, col, channels = 21, 21, 1
    num_classes = 2
    metricsFile = 'result.txt'
    kfold = 5
    x, y = load_ML_SL_EC_data()
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=59)
    (X_train_Kf, y_train_Kf), (X_test_Kf, y_test_Kf) = load_Kf_data(x, y, kfold=kfold, random_state=42)
    y_pred = np.zeros((0,num_classes))
    y_true = np.zeros((0,num_classes))
    
    for k in range(kfold):
        x_train = X_train_Kf[k].reshape((-1, row, col, channels))
        x_test = X_test_Kf[k].reshape((-1, row, col, channels))
        y_train = to_categorical(y_train_Kf[k], num_classes=num_classes)
        y_test = to_categorical(y_test_Kf[k], num_classes=num_classes)
        
        tf.keras.backend.clear_session()
        #model = CapsNet(input_shape=[row,col,channels], num_classes=num_classes, num_routing=5)
        model = resnet_v1(input_shape=(row, col, channels), depth=20, num_classes=num_classes)
        model.summary()
    
        model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
        
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
        
        model.fit(x_train, y_train,
              batch_size=50,
              epochs=20,
              validation_split=0.1,
              callbacks=[lr_decay])
    
        pred = model.predict(x_test, batch_size=50)
    
        noteInfo = "\n\n{}/{} cross-validation predict Multi-label EC and Single-label EC:\n".format(k, kfold)
        y_t = np.argmax(y_test, 1)
        y_p = np.argmax(pred, 1)
        cm = confusion_matrix(y_t, y_p)
        with open(metricsFile, 'a') as fw:
            fw.write(noteInfo)
            for i in range(2):
                fw.write(str(cm[i,0]) + '\t' + str(cm[i,1]) + '\n')
            fw.write("ACC:%f\n"%accuracy_score(y_t, y_p))
            fw.write("MCC:%f\n"%matthews_corrcoef(y_t, y_p))
        

        y_pred = np.concatenate((y_pred, pred))
        y_true = np.concatenate((y_true,y_test))
        
    noteInfo = "\nTotal cross-validation:\n"
    y_T = np.argmax(y_true, 1)
    y_P = np.argmax(y_pred, 1)
    cm = confusion_matrix(y_T, y_P)
    
    with open(metricsFile, 'a') as fw:
        fw.write(noteInfo)
        for i in range(2):
            fw.write(str(cm[i,0]) + '\t' + str(cm[i,1]) + '\n')
        fw.write("ACC:%f\n"%accuracy_score(y_T, y_P))
        fw.write("MCC:%f\n"%matthews_corrcoef(y_T, y_P))
    
    
            
if __name__ == "__main__":
    classify_slec_bi()
