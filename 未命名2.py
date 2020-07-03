# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:06:39 2020

@author: lwzjc
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.utils import shuffle
from resnet import resnet_v1
data=np.load('mlec_4479.npz')
#data=np.load('mlec_nr90.npz')
#data=np.load('pos_all_90_new.npz')
x,y = data['x'], data['y']
x,y = x[:4429], y[:4429,:6]
x = x.reshape((-1, 21*21*2))

def load_bottlenect_data(x, y, modelfile = './model/ec/weights-ec_259.h5'):        
    x,y = shuffle(x, y)
    
    tf.keras.backend.clear_session()
    base_model = resnet_v1(input_shape=(21, 21, 1), depth=20, num_classes=2)
    base_model.summary()
    base_model.load_weights(modelfile)
    model = models.Sequential()
    model.add(models.Model(inputs=base_model.input, outputs=base_model.get_layer('activation_18').output))
    model.add(layers.Flatten())
    model.summary()
    features = model.predict(x)
    #np.savez('mlec4479_bn_Features.npz', x=features, y=y)
    return features, y
'''
# Building a Label Graph
'''
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
edge_map = graph_builder.transform(y)
print("{} labels, {} edges".format(y.shape[1], len(edge_map)))

def to_membership_vector(partition):
    return{
        member: partition_id
        for partition_id, members in enumerate(partition)
        for member in members}

label_names = {'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6', 'EC7'}


# Using Networkx
from skmultilearn.cluster.networkx import NetworkXLabelGraphClusterer
clusterer = NetworkXLabelGraphClusterer(graph_builder, method='louvain')
partition = clusterer.fit_predict(x,y)
import networkx as nx

membership_vector = to_membership_vector(partition)
nx.draw(clusterer.graph_,
        pos=nx.circular_layout(clusterer.graph_),
        with_labels=True,
        width=[10*x/y.shape[0] for x in clusterer.weights_['weight']],
        node_color=[membership_vector[i] for i in range(y.shape[1])],
        cmap=plt.cm.Spectral,
        node_size=100,
        font_size=14)


'''
x = x[:,:,:,0]
x = x.reshape((-1,21,21,1))
x, y = load_bottlenect_data(x,y)
'''

'''
# Using iGraph
from skmultilearn.cluster import IGraphLabelGraphClusterer
#from skmultilearn.cluster import FixedLabelSpaceClusterer
import igraph as ig
#label_relations=[[0,1,2],[1,5],[3,4]]
#clusterer = FixedLabelSpaceClusterer(clusters=label_relations)

clusterer = IGraphLabelGraphClusterer(graph_builder=graph_builder, method='walktrap')
partition = clusterer.fit_predict(x, y)
colors=['red', 'white', 'blue']
membership_vector = to_membership_vector(partition)
visual_style = {
    "vertex_size" : 20,
    "vertex_label": [x[0] for x in label_names],
    "edge_width" : [10*x/y.shape[0] for x in clusterer.graph_.es['weight']],
    "vertex_color": [colors[membership_vector[i]] for i in range(y.shape[1])],
    "bbox": (400,400),
    "margin": 80,
    "layout": clusterer.graph_.layout_circle()

}
ig.plot(clusterer.graph_, **visual_style)
'''

from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.ensemble import RakelO
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier


classifier = LabelSpacePartitioningClassifier(
    classifier=LabelPowerset(classifier=RandomForestClassifier(n_estimators=100
                                                               ), 
                             require_dense=[True,True]),
    require_dense=[True, True],
    clusterer=clusterer)


'''
classifier = MajorityVotingClassifier(
    classifier=ClassifierChain(classifier=RandomForestClassifier(n_estimators=200)),
    clusterer=clusterer)
'''

'''
classifier = RakelO(
    base_classifier=RandomForestClassifier(n_estimators=100),
    base_classifier_require_dense=[True, True],
    labelset_size=2,
    model_count=7)
'''

# evaluate using 5-fold cross validation
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
#k_fold = IterativeStratification(n_splits=4428, order=1)
loo = LeaveOneOut()
result = []
i = 0
for train, test in loo.split(x,y):
    classifier.fit(x[train], y[train])
    predict = classifier.predict(x[test])
    result.append( accuracy_score(y[test], predict))
    i += 1
    if i%200==0:
        print("{}: {}".format(i, accuracy_score(y[test], predict)))

result = np.array(result)
print('mean: {}\nstd: {}\n'.format(np.mean(result), np.std(result)))
 


    