from scipy import ndimage
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import argparse
import os
import hickle


def main(params):
  
  batch_size = params['batch_size']
  max_length = params['max_length']
  word_count_threshold = params['word_count_threshold']
  vgg_model_path = params['vgg_model_path']

  splits = ['val', 'test']
  for split in splits:
    annotations = load_pickle('./data/%s/%s.annotations.pkl' %(split, split))

    if split == 'train':
      word_to_idx = build_word_to_idx(annotations['caption'], word_count_threshold)
      save_pickle(word_to_idx, './data/%s/word_to_idx.pkl' %split)
      captions = build_caption_vectors(annotations, word_to_idx, max_length)
      save_pickle(captions, './data/%s/%s.captions.pkl' %(split, split))

    file_names, id_to_idx = get_file_names(annotations)
    save_pickle(file_names, './data/%s/%s.file.names.pkl' %(split, split))

    if split == 'train':
      image_idxs = get_image_idxs(annotations, id_to_idx)
      save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' %(split, split))

    # Prepare reference captions to compute bleu scores later
    image_ids = {}
    feature_to_captions = {}
    i = -1
    for caption, image_id in zip(annotations['caption'], annotations['image_id']):
      if not image_id in image_ids:
        image_ids[image_id] = 0
        i += 1
        feature_to_captions[i] = []
      feature_to_captions[i].append(caption.lower() + ' .')
    save_pickle(feature_to_captions, './data/%s/%s.references.pkl' %(split, split))

  # Extract conv5_3 feature vectors
  vggnet = Vgg19(vgg_model_path)
  vggnet.build()
  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for split in splits:
      anno_path = './data/%s/%s.annotations.pkl' %(split, split)
      save_path = './data/%s/%s.features.hkl' %(split, split)
      annotations = load_pickle(anno_path)
      image_list = list(annotations['file_name'].unique())
      if split == 'train':
        image_path = map(lambda x: os.path.join('./image/train2014_resized', str(x)), image_list)
      else:
        image_path = map(lambda x: os.path.join('./image/val2014_resized', str(x)), image_list)
      n_examples = len(image_path)

      all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

      for start, end in zip(range(0, n_examples, batch_size), range(batch_size, n_examples+batch_size, batch_size)):
        image_batch_file = image_path[start:end]
        image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(np.float32)  

        feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
        all_feats[start:end, :] = feats
        print ("Processed %d %s features.." %(end, split))

      # Normalize feature vectors
      all_feats = np.reshape(all_feats, [-1, 512])
      mean = np.mean(all_feats, 0)
      var = np.var(all_feats, 0)
      all_feats = (all_feats - mean) / np.sqrt(var)
      all_feats = np.reshape(all_feats, [-1, 196, 512])

      # Use hickle to save huge numpy array
      hickle.dump(all_feats, save_path)
      print ("Saved %s.." %(save_path))
      

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', required=True, type=int, help='batch size when extracting feature vectors from VGGNet.')
  parser.add_argument('--max_length', default=15, type=int, help='max length of a caption, in number of words. longer than this clipped.')
  parser.add_argument('--word_count_threshold', default=3, type=int, help='clip words if occur less than this number in training dataset')
  parser.add_argument('--vgg_model_path', default='./data/imagenet-vgg-verydeep-19.mat', help='path for pretrained vggnet19 mat file')

  args = parser.parse_args()
  params = vars(args) 
  main(params)