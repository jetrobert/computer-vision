# -×-coding: utf-8-*-

import struct
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import sys

input_path = sys.argv[1] #mnist数据库解压后的所在路径
output_path = sys.argv[2] #生成的图片所在的路径

# =====read labels=====
label_file = input_path + '/t10k-labels.idx1-ubyte'
label_fp = open(label_file, 'rb')
label_buf = label_fp.read()

label_index=0
label_magic, label_numImages = struct.unpack_from('>II', label_buf, label_index)
label_index += struct.calcsize('>II')
labels = struct.unpack_from('>600B', label_buf, label_index)

# =====read train images=====
label_map = {}
train_file = input_path + '/t10k-images.idx3-ubyte'
train_fp = open(train_file, 'rb')
buf = train_fp.read()

index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')
k = 0
for image in range(0,numImages):
    label = labels[k]
    if(label_map.has_key(label)):
        ids = label_map[label] + 1
        label_map[label] += 1

    else:
        label_map[label] = 0
        ids = 0
    k += 1
    if(label_map[label] > 50):
            continue
    im=struct.unpack_from('>27B',buf,index)
    index+=struct.calcsize('>27B')

    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)
    #fig=plt.figure()
    #plotwindow=fig.add_subplot(111)
    #plt.imshow(im,cmap='gray')
    #plt.show()
    im=Image.fromarray(im)
    im.save(output_path + '/%s_%s.bmp'%(label, ids),'bmp')
    
    
