# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:29:32 2020

@author: lwzjc
"""
from resnet import resnet_v1
from prepareDataset import load_mlec_nr
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from skmultilearn.adapt import MLkNN, MLTSVM
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.utils import shuffle
'''
x, y = load_mlec_nr(nr=90)
x,y = shuffle(x, y)
modelfile = './model/ec/weights-ec_259.h5'
tf.keras.backend.clear_session()
base_model = resnet_v1(input_shape=(21, 21, 1), depth=20, num_classes=2)
base_model.summary()
base_model.load_weights(modelfile)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
features = model.predict(x)
np.savez('mlecFeatures.npz', x=features, y=y)
'''

data = np.load('mlecFeatures.npz')
x, y = data['x'], data['y']
'''
#parameters = {'k': range(5,13), 's': [0.5, 0.7, 1.0]}#{'k': 12, 's': 0.5} 0.2609881275890115
parameters = {'c_k': [2**i for i in range(-5, 5, 2)]}
score = 'accuracy'

clf = GridSearchCV(MLTSVM(), parameters, scoring=score)
clf.fit(features, y)

print (clf.best_params_, clf.best_score_)
'''

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset,ClassifierChain
from sklearn.ensemble import RandomForestClassifier
import time
import xgboost as xgb
from tensorflow.keras.utils import to_categorical
from collections import Counter
#rf = RandomForestClassifier(n_estimators=500,class_weight='balanced')
start=time.time()
#clf = BinaryRelevance(classifier = RandomForestClassifier(), require_dense = [True, True])
#clf = LabelPowerset(classifier=rf, require_dense=[True,True])
#clf = ClassifierChain(classifier=rf, require_dense=[True, True])


score = 'balanced_accuracy'

#result = cross_validate(rf, x, y[:,6], cv=3, scoring=score,return_train_score=True)


print('training time taken: ',round(time.time()-start,0),'seconds')

dtrain = xgb.DMatrix(x[:-700],y[:-700])
dtest = xgb.DMatrix(x[-700:6], y[-700:6])
