# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 07:54:52 2020

@author: lwzjc
"""
import tensorflow as tf

row, col, chanel = 50, 20, 10
    
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# conv1 layer
with tf.name_scope('Input_Layer'):
    x = tf.placeholder('float', shape=[None,row,col,chanel])

def net(x_input):
    with tf.name_scope('conv1'):
        W1 = weight([5,5,1,128])
        b1 = bias([128])
        conv1 = conv2d(x_input, W1) + b1
        conv1 = tf.nn.relu(conv1)
    with tf.name_scope('pool1'):
        pool1 = max_pool_2x2(conv1)
    
    # conv2 layer
    with tf.name_scope('conv2'):
        W2 = weight([5,5,128, 256])
        b2 = bias([256])
        conv2 = conv2d(pool1, W2) + b2
        conv2 = tf.nn.relu(conv2)
    with tf.name_scope('pool2'):
        pool2 = max_pool_2x2(conv2)
        
    # flatten layer
    with tf.name_scope("flat"):
        p = pool2.shape
        d = p[1] * p[2] * p[3]
        flat1 = tf.reshape(pool2, [-1, ])
    
    # hidden layer
    with tf.name_scope('hidden'):
        W3 = weight([d, 512])
        b3 = bias([512])
        hidden = tf.nn.relu(tf.matmul(flat1, W3) + b3)
        drop = tf.nn.dropout(hidden, keep_prob=0.8)
    
    # output layer
    with tf.name_scope('out'):
        W4 = weight([512, 7])
        b4 = bias([7])
        y_pred = tf.nn.softmax(tf.matmul(drop, W4) + b4)
    
    return y_pred

def loss_fun(x):
    y = 
    for i in range(chanel):
        y = net(x[:,:,:,chanel])
        
        loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            logits=y_pred, labels=y_label))
# optimizer
with tf.name_scope('optimizer'):
    y_label = tf.placeholder("float", shape=[None, 7], name='y_label')
    loss_function = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            logits=y_pred, labels=y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)
    
# Training parameter
trainEpochs = 30
batchSize = 32
totalBatchs = x_train.shape[0]//batchSize 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training ......
for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        start = i * batchSize
        if epoch + 1 == totalBatchs:
            batch_x, batch_y = x_train[start:], y_train[start:]
        else:
            end = (epoch+1) * batchSize
            batch_x, batch_y = x_train[start:end], y_train[start:end]
        
        sess.run(optimizer, feed_dict={x:batch_x, y_label: batch_y})
            