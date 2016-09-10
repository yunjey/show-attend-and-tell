import tensorflow as tf
import numpy as np
import cPickle as pickle
import hickle 
import time
import os


def init_weight(name, shape, stddev=1.0, dim_in=None):
    if dim_in is None:
        dim_in = shape[0] 
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev/np.sqrt(dim_in)), name=name)

def init_bias(name, shape):
    return tf.Variable(tf.zeros(shape), name=name)

def load_coco_data(data_path='./data', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}
    
    data['features'] = hickle.load(os.path.join(data_path, '%s.features.hkl' %split))
    with open(os.path.join(data_path, '%s.captions.pkl' %split), 'rb') as f:
        data['captions'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.image.idxs.pkl' %split), 'rb') as f:
        data['image_idxs'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.file.names.pkl' %split), 'rb') as f:
        data['file_names'] = pickle.load(f)                    
    with open(os.path.join(data_path, '%s.annotations.pkl' %split)     , 'rb') as f:
        data['annotations'] = pickle.load(f)          
    if split == 'train':
        with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
            data['word_to_idx'] = pickle.load(f)
            
    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        else:
            print k, type(v), len(v)
    end_t = time.time()
    print "elapse time: %.2f" %(end_t - start_t)
    return data

def decode_captions(captions, idx_to_word):
    """
    Inputs:
    - captions: numpy ndarray where the value(word index) is in the range [0, V) of shape (N, T)
    - idx_to_word: index to word mapping dictionary

    Returns:
    - decoded: list of length N,
    """
    N, T = captions.shape
    decoded = []

    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    return decoded

def sample_coco_minibatch(data, batch_size):
    """
    Inputs: 
    - data: dictionary with following keys:
        - train_features: ndarray of shape (82783, 196, 512), not yet completed
        - train_image_filename: list of length 82783
        - train_captions: ndarray of shape (410000, 17)
        - train_image_idxs: ndarray of shape (410000,)
        - val_features (will be added)
        - val_image_filename (will be added)
        - val_captions (will be added)
        - val_image_idxs (will be added)
    - batch_size: batch size
    - split: train or val
    """

    data_size = data['captions'].shape[0]
    mask = np.random.choice(data_size, batch_size)
    captions = data['captions'][mask]
    image_idxs = data['image_idxs'][mask]
    features = data['features'][image_idxs]
    image_files = data['file_names'][image_idxs]

    return captions, features, image_files
