import tensorflow as tf
import numpy as np
import cPickle as pickle
import hickle
import time
import os

def init_weight(name, shape, stddev=1.0, dtype=tf.float32, dim_in=None):
    if dim_in is None:
        dim_in = shape[0] 
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev/np.sqrt(dim_in)), name=name, dtype=dtype)

def init_bias(name, shape, dtype=tf.float32):
    return tf.Variable(tf.zeros(shape), name=name, dtype=dtype)

def load_coco_data(data_path='./data', split='train', feature='conv5_3'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}
    
    if feature == 'conv5_3':
        # tensorflow conv5_3 features
        data['features'] = hickle.load(os.path.join(data_path, '%s.features.conv5_3.hkl' %split))
    elif feature == 'conv5_4':
        # tensorflow conv5_4 features
        data['features'] = hickle.load(os.path.join(data_path, '%s.features.conv5_4.hkl' %split))
    elif feature == 'sparse':
         # theano conv5_3 sparse features
        data['features'] = hickle.load(os.path.join(data_path, '%s.features.conv5_3.sparse.hkl' %split))
    with open(os.path.join(data_path, '%s.captions.pkl' %split), 'rb') as f:
        data['captions'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.image.idxs.pkl' %split), 'rb') as f:
        data['image_idxs'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.file.names.pkl' %split), 'rb') as f:
        data['file_names'] = pickle.load(f)            
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
    - captions: numpy ndarray which contains word indices in the range [0, V), of shape (N, T)
    - idx_to_word: index to word mapping dictionary
    Returns:
    - decoded: decoded senteces; list of length N
    """
    N, T = captions.shape
    decoded = []

    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def sample_coco_minibatch(data, batch_size):
    """
    Inputs: 
    - data: dictionary with following keys:
        - captions: caption vectors of shape (25000, 17)
        - image_idxs: indices mapping from captions to features of shape (25000,)
        - features: feature vectors of shape (5000, 196, 512)
        - file_names: image file names of (5000,)
    - batch_size: mini-batch size
    """
    data_size = data['captions'].shape[0]
    mask = np.random.choice(data_size, batch_size)
    captions = data['captions'][mask]
    image_idxs = data['image_idxs'][mask]
    features = data['features'][image_idxs]
    image_files = data['file_names'][image_idxs]

    return captions, features, image_files