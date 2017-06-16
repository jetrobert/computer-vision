import cPickle 

def main(data_size, path):   
    label_names = ['WalkingSlowly', 'WalkingQuickly', 'JogingSlowly', 'JogingQuickly', \
				'DownstairsSlowly', 'UpstairsSlowly', 'DownstairsQuickly', 'UpstairsQuickly', \
				'HalfSitDownQuickly', 'HalfSitUpQuickly', 'LowSitDownQuickly', 'LowSitUpQuickly', \
				'LyLefttoSitSlowly', 'SitToLyLeftSlowly', 'LyLefttoSitQuickly', 'SittoLyLeftQuickly', \
				'LyBack', 'LyBacktoLateral', 'LyLateral', 'LyLateraltoBack', 'Bending', \
				'BendingtoStanding', 'Standing', 'StandtoBending', 'BendDown', 'BendUp', \
				'Stumble', 'Walking', 'Jump', 'LyLeft', 'Sit']  
    num1 = 1800  
    num2 = data_size * data_size * 3  
    dic = {'num_cases_per_batch': num1, 'label_names':label_names, 'num_vis':num2}  
    out_file = open(path+'/batches.meta', 'w+')  
    cPickle.dump(dic, out_file)  
    out_file.close()  
	
if __name__ == '__main__':
    data_size = 128
    path = '../dataset/dataset-batches-py'
    main(data_size, path)