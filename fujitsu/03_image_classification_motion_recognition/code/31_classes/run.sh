#!/bin/bash

# preprocess dataset
dataset_rename.py
dataset_partition.py
img_resize.py
img2bin_train.py
img2bin_test.py
meta_info.py

# train with cnn
cnn_train.py

# test with cnn
cnn_test.py
