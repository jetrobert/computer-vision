# -*- coding: utf-8 -*-

""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.

Applying VGG 16-layers convolutional network to CIFAR-10 Dataset classification task.

References:
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    K. Simonyan, A. Zisserman. arXiv technical report, 2014.

Links:
    - http://arxiv.org/pdf/1409.1556
	- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()
Y = tflearn.data_utils.to_categorical(Y, 10)
testY = tflearn.data_utils.to_categorical(testY, 10)

# Building 'VGG Network'
network = input_data(shape=[None, 32, 32, 3])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Training
model = tflearn.DNN(network, checkpoint_path='checkpoint_cifar10/model_vgg',
                    max_checkpoints=10, tensorboard_verbose=0)
					
model.fit(X, Y, n_epoch=100, validation_set=(testX, testY), shuffle=True,
          show_metric=True, batch_size=128, snapshot_step=500,
          snapshot_epoch=False, run_id='vgg_cifar10')
		  
model.save('model/cifar10_retrained_by_vgg16')
