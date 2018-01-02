# -*-coding:utf-8-*-
import numpy
import theano
from PIL import Image
from pylab import *
import os
import theano.tensor as T
import random
import pickle


def dataresize(path=r'train'):
    # test path
    path_t =r"test"
    # train path
    datas = []
    train_x= []
    train_y= []
    valid_x= []
    valid_y= []
    test_x= []
    test_y= []
    for dirs in os.listdir(path):
        # print dirs
        for filename in os.listdir(os.path.join(path,dirs)):
            imgpath =os.path.join(os.path.join(path,dirs),filename)
            img = Image.open(imgpath)
            img =img.convert('L').resize((28,28))
            width,hight=img.size
            img = numpy.asarray(img,dtype='float64')/256.

            tmp = img.reshape(1, hight*width)[0]
            tmp =hstack((dirs,tmp))  # 在此将标签加在数据的前面。

            datas.append(tmp)
       # datas.append(img.reshape(1, hight*width)[0])
        #在此处取出第一行的数据否则在后面的转换的过程中会出现叠加的情况，在成在转换成矩阵时宝类型转换的错误
    #将数据打乱顺序
    random.shuffle(datas)
    # 将数据和标签进行分离
    label=[]
    for num in range(len(datas)):
        label.append((datas[num])[0])
        datas[num] =(datas[num])[1:]
    #将数据的标签项去掉
    '''
    tests = []
    # #读取测试集
    for dirs in os.listdir(path_t):
        for filename in os.listdir(os.path.join(path_t,dirs)):
            imgpath =os.path.join(os.path.join(path_t,dirs),filename)
            img = Image.open(imgpath)
            img =img.convert('L').resize((28,28))
            width,hight=img.size
            img = numpy.asarray(img,dtype='float64')/256.
            tmp = img.reshape(1, hight*width)[0]
            # 在此如果不是取出[0]的话在后面会发现其实其是一个多维的数据的叠加，
            # 在后面使用theano中的cnn在调用时会出现数据的异常（转换的异常），
            # 在此是跟原始的mnist的数据集的形式做了比较修改才发现的。。。
            tmp =hstack((dirs,tmp))
            tests.append(tmp)
    #将数据打乱顺序
    random.shuffle(tests)
    #  将数据和标签进行分离
    label_t=[]
    for num in range(len(tests)):
        label_t.append((tests[num])[0])
        tests[num] =(tests[num])[1:]
    #将数据的标签项去掉
    '''
    '''    将数据进行打乱，拆分成train test valid    '''
    for num in range(len(label)):
        train_x.append(datas[num])
        train_y.append(label[num])
    '''
    for num in range(len(tests)):
        if num%2==0:
            valid_x.append(tests[num])
            valid_y.append(label_t[num])
        if num%2==1:
            test_x.append(tests[num])
            test_y.append(label_t[num])
    '''
    train_x=numpy.asarray(train_x,dtype='float64')
    train_y=numpy.asarray(train_y,dtype='int64')
    '''
    valid_x=numpy.asarray(valid_x,dtype='float64')
    valid_y=numpy.asarray(valid_y,dtype='int64')
    test_x=numpy.asarray(test_x,dtype='float64')
    test_y=numpy.asarray(test_y,dtype='int64')
    '''
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    '''
    test_set_x, test_set_y = shared_dataset((test_x,test_y))
    valid_set_x, valid_set_y = shared_dataset((valid_x,valid_y))
    '''
    train_set_x, train_set_y = shared_dataset((train_x,train_y))
    #rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    rval = [(train_set_x, train_set_y)]
    save_datas_pkl(rval)
        #  return rval
def save_datas_pkl(file1s,path=r'./data/train-datasets.ubyte'):
    datas=file1s
    output =open(path,'wb')
    pickle.dump(datas,output)
    output.close()
def load_datas(path=r'./data/train-datasets.ubyte'):
    pkl_file =open(path,'rb')
    datas =pickle.load(pkl_file)
    pkl_file.close()
    return datas

if __name__=='__main__':
    dataresize()
