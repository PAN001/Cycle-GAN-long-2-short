import scipy.io
import scipy.misc
import random
import os

try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk


# "A valid path is the path to a directory that contains a directories that contains photo objects"
PATH1 = 'D:/In-shop Clothes Retrieval Benchmark/Img/img/img/WOMEN/Shorts/'
PATH2 = 'D:/In-shop Clothes Retrieval Benchmark/Img/img/img/WOMEN/Leggings/'


SAVE_PATH1 = 'D:/In-shop Clothes Retrieval Benchmark/Img/img/img/WOMEN/Shorts/All'
SAVE_PATH2 = 'D:/In-shop Clothes Retrieval Benchmark/Img/img/img/WOMEN/Leggings/All'

def trim_directory(path):
	entries = os.listdir(path)
	valid_entries = []
	for entry in entries:
		if 'All' not in entry:
			valid_entries.append(path + entry + '/')
	return valid_entries

pathes1 = trim_directory(PATH1)
pathes2 = trim_directory(PATH2)

def save_image(open_path, save_path, save_name):
	my_image = scipy.misc.imread(open_path)
	save_path = save_path + '/'+save_name+'.jpg'
	scipy.misc.imsave(save_path, my_image)

def read_path(path, save_path, step):
	for file in scandir(path):
		if file.name.endswith('.jpg') and file.is_file():
			thispath = file.path
			save_image(thispath, save_path, str(step))
			step = step + 1
	return step

def load_path(pathes, save_path):
	step = 0
	for path in pathes:
		step = read_path(path, save_path, step)


load_path(pathes1, SAVE_PATH1)
load_path(pathes2, SAVE_PATH2)



