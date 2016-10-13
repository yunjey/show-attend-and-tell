import tensorflow as tf
import sys
sys.path.append('../core')
import os
import hickle
from prepare_data import *
from core.vggnet import VGGNet 


def main(arg):
	"""
		e.g. arg = ['imagenet-vgg-verydeep-19.mat', 'conv5_3', '100']
	"""
	model = VGGNet(arg[0])
	assert arg[1] in ['conv5_3', 'conv5_4']
	conv5_3 = model.build_model(arg[1])
	batch_size = int(arg[2])
	splits = ['val', 'test', 'train']

	with tf.Session() as sess:
		for split in splits:
			anno_path = '../data/%s/%s.annotations.pkl' %(split, split)
			save_path = '../data/%s/%s.features.hkl' %(split, split)
			annotations = load_pickle(anno_path)
      		image_list = list(annotations['file_name'].unique())
      		image_path = map(lambda x: os.path.join('../image/', str(x)), image_list)
      		n_examples = len(image_path)

			all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

			for start, end in zip(range(0, n_examples, batch_size), 
					range(batch_size, n_examples+batch_size, batch_size)):
				image_batch_file = image_path[start:end]
	      		image_batch = np.array(map(lambda x: 
	           		ndimage.imread(x, mode='RGB'), image_batch_file)).astype(np.float32)  

	      		feats = sess.run(conv5_3, feed_dict={model.images: image_batch})
	      		all_feats[start:end, :] = feats.reshape(-1, 196, 512)
	      		print str(start + batch_size) + " finished.. "

	      	# using hickle to save huge numpy array
	      	hickle.dump(all_feats, save_path)
	      	print "save %s.." %(save_path)



if __name__ == "__main__":
	main(sys.argv[1:])
