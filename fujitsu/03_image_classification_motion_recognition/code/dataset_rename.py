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
	label_names = ['D01WalkingSlowly', 'D02WalkingQuickly', 'D03JogingSlowly', 'D04JogingQuickly', \
				'D05DownstairsSlowly', 'D05UpstairsSlowly', 'D06DownstairsQuickly', 'D06UpstairsQuickly', \
				'D08HalfSitDownQuickly', 'D08HalfSitUpQuickly', 'D10LowSitDownQuickly', 'D10LowSitUpQuickly', \
				'D12LyLefttoSitSlowly', 'D12SitToLyLeftSlowly', 'D13LyLefttoSitQuickly', 'D13SittoLyLeftQuickly', \
				'D14LyBack', 'D14LyBacktoLateral', 'D14LyLateral', 'D14LyLateraltoBack', \
				'D15Bending', 'D15BendingtoStanding', 'D15Standing', 'D15StandtoBending', \
				'D16BendDown', 'D16BendUp', 'D18Stumble', 'D18Walking', \
				'D19Jump', 'D1213LyLeft', 'D1213Sit']
	for label_num in range(len(label_names)):
		path = '../dataset/dataset_process/'+label_names[label_num]+'/'
		file_rename(path, label_num)