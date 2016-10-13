import tensorflow as tf
from vggnet import VGGNet

model = VGGNet('imagenet-vgg-verydeep-19.mat')
conv5_3 = model.build_model('conv5_3')


splits = ['val', 'test', 'train']



import cPickle as pickle

def load_pickle(path):
	with open(path, 'rb') as f:
		pkl = pickle.load(f)
		print 'loaded %s..' %path
		return pkl  

def save_pickle(data, path):
	with open(path, 'wb') as f:
		pickle.dump(data, path, pickle.HIGHEST_PROTOCOL)
		print 'saved %s..' %path




with tf.Session() as sess:
	for split in splits:
		anno_path = '%s/%s.annotations.pkl' %(split, split)
		save_path = '%s/%s.features.pkl' %(split, split)


