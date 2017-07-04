import os
import random
import shutil

def file_move(path, path_i):
	filenames = os.listdir(path)
	#batch_num = len(filenames)//5
	batch_num = 1800
	# splilt the train data
	for i in range(batch_num):
		#print(name)
		name = filenames[i]
		if os.path.isfile(path+name):     
			#shutil.copy(path+name, path_target)
			shutil.move(path+name, path_i)
			print "move %s to %s successful" % (name, path_i)
			
if __name__ == '__main__':
	path = '../dataset/dataset_train2/'
	path_train = '../dataset/dataset_trains'
	for i in range(5):
		path_i = path_train+'/train_'+str(i+1)
		path_exist = os.path.exists(path_i)
		if not path_exist:
			os.makedirs(path_i)
		file_move(path, path_i)