# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import operator
from os import listdir
import sys
import cPickle as pickle
import random
import os

data={}
list1=[]
list2=[]
list3=[]
pixel=256

def img_trans(start,end):
    for k in range(start,end):
        currentpath=folder+"/"+imglist[k]
        im=Image.open(currentpath)
        #width=im.size[0]
        #height=im.size[1]
        x_s=pixel
        y_s=pixel
        out = im.resize((x_s,y_s),Image.ANTIALIAS)
        out.save(folder+"/"+str(imglist[k]))
	
def seplabel(fname):
    filestr=fname.split(".")[0]
    label=int(filestr.split("_")[0])
    return label

def main(start,end):
    global data
    global list1
    global list2
    global list3
    global pixel
    for k in range(start,end):
        currentpath=folder+"/"+imglist[k]
        im=Image.open(currentpath)
        with open(binpath, 'a') as f:
            for i in range (0,pixel):
                for j in range (0,pixel):
                    cl=im.getpixel((i,j))
                    list1.append(cl[0])
            for i in range (0,pixel):
                for j in range (0,pixel):
                    cl=im.getpixel((i,j))
                    #with open(binpath, 'a') as f:
                    #mid=str(cl[1])
                    #f.write(mid)
                    list1.append(cl[1])
            for i in range (0,pixel):
                for j in range (0,pixel):
                    cl=im.getpixel((i,j))
                    list1.append(cl[2])
        list2.append(list1)
        list1=[]
        f.close()
        print("image"+str(k+1)+"saved.")
        list3.append(imglist[k].encode('utf-8'))
    arr2=np.array(list2,dtype=np.uint8)
    data['batch_label'.encode('utf-8')]='testing batch 1 of 1'.encode('utf-8')
    data.setdefault('labels'.encode('utf-8'),label)
    data.setdefault('data'.encode('utf-8'),arr2)
    data.setdefault('filenames'.encode('utf-8'),list3)
    output = open(binpath, 'wb')
    pickle.dump(data, output)
    output.close()

if __name__=="__main__":
    folder="../../dataset/10_classes/test_img_resized_256"
    dir_path="../../dataset/10_classes/dataset-batches-py-256"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    imglist=listdir(folder)
    nums=len(imglist)
    start=0
    end=nums
    img_trans(start,end)
    label=[]
    for i in range (start,end):
        label.append(seplabel(imglist[i]))
    binpath="../../dataset/10_classes/dataset-batches-py-256/test_batch"
    if not os.path.exists(binpath):
        os.system(r'touch %s' % binpath)
    print(binpath)
    main(start,end)
	