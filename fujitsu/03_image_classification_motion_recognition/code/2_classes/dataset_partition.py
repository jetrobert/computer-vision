import os
import random
import shutil

def file_move(path, path_train, path_test, label_num):
	filenames = os.listdir(path)
	test_num = len(filenames)//6
	random_num = random.sample(range(10,len(filenames)),test_num)
	
	# randomly choose test data
	for num_ in random_num:
		file = path+str(label_num)+'_'+str(num_)+'.jpg'
		if not os.path.isdir(path_test):
			os.makedirs(path_test)
		shutil.move(file, path_test)
		print "move %s to %s successful" % (file, path_test)
		
	# the rest is train data
	for name in filenames:
		#print(name)
		if os.path.isfile(path+name):     
			#shutil.copy(path+name, path_target)
			if not os.path.isdir(path_train):
				os.makedirs(path_train)
			shutil.move(path+name, path_train)
			print "move %s to %s successful" % (name, path_train)

def partiton(path_train, path_test):		
	# make train:test = 6:1
	file_train_num = len(os.listdir(path_train))
	file_test_num = len(os.listdir(path_test))
	file_total_num = file_train_num + file_test_num
	if file_total_num > file_test_num * 6:
		for i in range((file_total_num-file_test_num*6)/6):
			shutil.move(path_train+'/'+os.listdir(path_train)[i], path_test)
	elif file_total_num < file_test_num * 6:
		for i in range((file_test_num*6-file_total_num)/6):
			shutil.move(path_test+'/'+os.listdir(path_test)[i], path_train)
			
if __name__ == '__main__':
	label_names = ['dynamic', 'static']
	for label_num in range(len(label_names)):
		path = '../../dataset/2_classes/dataset_rename/'+label_names[label_num]+'/'
		path_train = '../../dataset/2_classes/dataset_train'
		path_test = '../../dataset/2_classes/dataset_test'
		file_move(path, path_train, path_test, label_num)
	partiton(path_train, path_test)
		