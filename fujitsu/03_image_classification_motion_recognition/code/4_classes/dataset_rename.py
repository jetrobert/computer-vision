import os

def file_rename(path, label_num):
	filenames = os.listdir(path)
	for name in filenames:
		print(name)
	for num in range(0,len(filenames)):
		if(num<10):
			print()
			print(filenames[num])
			os.rename(path+filenames[num], path+str(label_num)+'_'+'0'+str(num)+'.jpg')
		else:
			os.rename(path+filenames[num], path+str(label_num)+'_'+str(num)+'.jpg')
			
if __name__ == '__main__':
	label_names = ['D02WalkingQuickly', 'D04JogingQuickly', 
				   'D05DownstairsSlowly', 'D05UpstairsSlowly']
	for label_num in range(len(label_names)):
		path = '../../dataset/4_classes/dataset_process/'+label_names[label_num]+'/'
		file_rename(path, label_num)