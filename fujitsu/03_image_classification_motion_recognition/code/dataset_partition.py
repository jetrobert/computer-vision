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
		print "remove %s to %s successful" % (file, path_test)
		
	# the rest is train data
	for name in filenames:
		#print(name)
		if os.path.isfile(path+name):     
			#shutil.copy(path+name, path_target)
			shutil.move(path+name, path_train)
			print "remove %s to %s successful" % (name, path_train)
			
if __name__ == '__main__':
	label_names = ['D01WalkingSlowly', 'D02WalkingQuickly', 'D03JogingSlowly', 'D04JogingQuickly', \
				'D05DownstairsSlowly', 'D05UpstairsSlowly', 'D06DownstairsQuickly', 'D06UpstairsQuickly', \
				'D08HalfSitDownQuickly', 'D08HalfSitUpQuickly', 'D10LowSitDownQuickly', 'D10LowSitUpQuickly', \
				'D12LyLefttoSitSlowly', 'D12SitToLyLeftSlowly', 'D13LyLefttoSitQuickly', 'D13SittoLyLeftQuickly', \
				'D14LyBack', 'D14LyBacktoLateral', 'D14LyLateral', 'D14LyLateraltoBack', \
				'D15Bending', 'D15BendingtoStanding', 'D15Standing', 'D15StandtoBending', \
				'D16BendDown', 'D16BendUp', 'D18Stumble', 'D18Walking', \
				'D19Jump', 'D1213LyLeft', 'D1213Sit']
	for label_num in range(len(label_names)):
		path = '../dataset/dataset_p/'+label_names[label_num]+'/'
		path_train = '../dataset/dataset_train'
		path_test = '../dataset/dataset_test'
		file_move(path, path_train, path_test, label_num)