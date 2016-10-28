import tensorflow as tf
import numpy as np
import cPickle as pickle
import hickle
import time
import os

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
  if split == 'train':
    with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
      data['word_to_idx'] = pickle.load(f)
          
  for k, v in data.iteritems():
    if type(v) == np.ndarray:
      print k, type(v), v.shape, v.dtype
    else:
      print k, type(v), len(v)
  end_t = time.time()
  print "Elapse time: %.2f" %(end_t - start_t)
  return data

def decode_captions(captions, idx_to_word):
  if captions.ndim == 1:
    T = captions.shape[0]
    N = 1
  else:
    N, T = captions.shape

  decoded = []
  for i in range(N):
    words = []
    for t in range(T):
      if captions.ndim == 1:
        word = idx_to_word[captions[t]]
      else:
        word = idx_to_word[captions[i, t]]
      if word == '<END>':
        words.append('.')
        break
      if word != '<NULL>':
        words.append(word)
    decoded.append(' '.join(words))
  return decoded

def sample_coco_minibatch(data, batch_size):
  data_size = data['captions'].shape[0]
  mask = np.random.choice(data_size, batch_size)
  captions = data['captions'][mask]
  image_idxs = data['image_idxs'][mask]
  features = data['features'][image_idxs]
  image_files = data['file_names'][image_idxs]

  return captions, features, image_files


def write_bleu(scores, path, epoch):
  if epoch == 0:
    file_mode = 'w'
  else:
    file_mode = 'a'
  with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
    f.write('Epoch %d\n' %(epoch+1))
    f.write('Bleu_1: %f\n' %scores['Bleu_1'])
    f.write('Bleu_2: %f\n' %scores['Bleu_2'])
    f.write('Bleu_3: %f\n' %scores['Bleu_3'])  
    f.write('Bleu_4: %f\n' %scores['Bleu_4']) 
    f.write('METEOR: %f\n' %scores['METEOR'])  
    f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])  
    f.write('CIDEr: %f\n\n' %scores['CIDEr'])

def load_pickle(path):
  with open(path, 'rb') as f:
    file = pickle.load(f)
    print ('Loaded %s..' %path)
    return file  

def save_pickle(data, path):
  with open(path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print ('Saved %s..' %path)

def build_word_to_idx(sentences, threshold=3): 
  word_counts = {}
  n_sents = 0
  for i, sentence in enumerate(sentences):
    n_sents += 1
      
    for word in sentence.lower().split(' '):
      word_counts[word] = word_counts.get(word, 0) + 1

  vocab = [word for word in word_counts if word_counts[word] >= threshold]
  print ('Filtered words from %d to %d' % (len(word_counts), len(vocab)))

  word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
  idx = 3
  for word in vocab:
    word_to_idx[word] = idx
    idx += 1
  return word_to_idx

def build_caption_vectors(annotations, word_to_idx, max_len=15):
  n_examples = len(annotations)
  captions = np.ndarray((n_examples,max_len+2)).astype(np.int32)   

  for i, sentence in enumerate(annotations['caption']):
    words = sentence.lower().split(' ')
    n_words = len(words)
    capvec = []
    
    capvec.append(word_to_idx['<START>'])
    for word in words:
      if word in word_to_idx:
        capvec.append(word_to_idx[word])
    capvec.append(word_to_idx['<END>'])

    if len(capvec) < (max_len + 2):
      for j in range(max_len + 2 - len(capvec)):
        capvec.append(word_to_idx['<NULL>']) 
    captions[i, :] = np.asarray(capvec)
  return captions

def get_file_names(annotations):
  image_file_names = []
  id_to_idx = {}
  idx = 0
  image_ids = annotations['image_id']
  file_names = annotations['file_name']

  for image_id, file_name in zip(image_ids, file_names):
    if not image_id in id_to_idx:
      id_to_idx[image_id] = idx
      image_file_names.append(file_name)
      idx += 1
  
  file_names = np.asarray(image_file_names)
  return file_names, id_to_idx

def get_image_idxs(annotations, id_to_idx):
  image_idxs = np.ndarray(len(annotations), dtype=np.int32)
  image_ids = annotations['image_id']
  for i, image_id in enumerate(image_ids):
    image_idxs[i] = id_to_idx[image_id]
  return image_idxs