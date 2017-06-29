import os
import cPickle 

def main(data_size, path):   
    label_names = ['dynamic', 'static'] 
    num1 = 2000
    num2 = data_size * data_size * 3  
    dic = {'num_cases_per_batch': num1, 'label_names':label_names, 'num_vis':num2}
    meta_file = path+'/batches.meta'
    if not os.path.exists(meta_file):
        os.system(r'touch %s' % meta_file)
    out_file = open(meta_file, 'w+')  
    cPickle.dump(dic, out_file)  
    out_file.close()  
	
if __name__ == '__main__':
    data_size = 256
    path = '../../dataset/2_classes/dataset-batches-py-256'
    main(data_size, path)
