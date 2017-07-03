import os
import skimage.io
from skimage import transform


def sep_label(fname):
	filestr=fname.split(".")[0]
	label=int(filestr.split("_")[0])
	num=int(filestr.split("_")[1])
	return label, num

def img_resize(path, path_target):
	if not os.path.isdir(path_target):
		os.makedirs(path_target)
	imglist=os.listdir(path)
	for img_name in imglist:
		label,img_num=sep_label(img_name)
		img=skimage.io.imread(path+'/'+img_name)
		dst=transform.resize(img, (256, 256))
		skimage.io.imsave(path_target+'/'+str(label)+'_'+str(img_num)+'.png', dst)
	
if __name__=='__main__':
	path="../../dataset/10_classes/dataset_train"
	path_target="../../dataset/10_classes/train_img_resized_256"
	img_resize(path, path_target)
