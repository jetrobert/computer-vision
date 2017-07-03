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
		shutil.move(file, path_test)
		print "move %s to %s successful" % (file, path_test)
		
	# the rest is train data
	for name in filenames:
		#print(name)
		if os.path.isfile(path+name):     
			#shutil.copy(path+name, path_target)
			shutil.move(path+name, path_train)
			print "move %s to %s successful" % (name, path_train)
			
if __name__ == '__main__':
	label_names = ['D02WalkingQuickly', 'D04JogingQuickly', 'D06DownstairsQuickly', 
				   'D06UpstairsQuickly', 'D10LowSitDownQuickly', 'D10LowSitUpQuickly', 
				   'D14LyBack', 'D14LyLateral', 'D15Standing', 'D18Stumble', 'D19Jump', 
				   'D1213Sit']
	for label_num in range(len(label_names)):
		path = '../../dataset/12_classes/dataset_rename/'+label_names[label_num]+'/'
		path_train = '../../dataset/12_classes/dataset_train'
		path_test = '../../dataset/12_classes/dataset_test'
		file_move(path, path_train, path_test, label_num)