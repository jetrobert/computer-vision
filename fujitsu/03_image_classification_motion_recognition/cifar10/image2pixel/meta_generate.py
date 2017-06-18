import cPickle 

def main(data_size, path):   
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  
    num1 = 10000  
    num2 = data_size * data_size * 3  
    dic = {'num_cases_per_batch': num1, 'label_names':label_names, 'num_vis':num2}  
    out_file = open(path+'/batches.meta', 'w+')  
    cPickle.dump(dic, out_file)  
    out_file.close()  
	
if __name__ == '__main__':
    data_size = 32
    path = '../dataset/cifar-10-batches-gene'
    main(data_size, path)