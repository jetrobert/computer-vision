# -*- coding: utf-8 -*-

""" AlexNet.

Applying 'Alexnet' to CIFAR-10 Dataset classification task..

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""

from __future__ import division, print_function, absolute_import
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# cpu only
os.environ['CUDA_VISIBLE_DEVICES']=''

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()
Y = tflearn.data_utils.to_categorical(Y, 10)
testY = tflearn.data_utils.to_categorical(testY, 10)

# Building 'AlexNet'
network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='checkpoint_cifar10/model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=200, validation_set=(testX, testY), shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_cifar10')
		  
model.save('model/cifar10_retrained_by_alexnet')
