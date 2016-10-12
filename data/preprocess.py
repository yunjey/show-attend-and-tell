import numpy as np


def build_word_to_idx(sentences, threshold=5): 
    word_counts = {}
    n_sents = 0
    max_len = 0
    for i, sentence in enumerate(sentences):
        n_sents += 1
        
        if len(sentence.split(" ")) > max_len:
            max_len = len(sentence.split(" "))
            
        for word in sentence.lower().split(' '):
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab = [word for word in word_counts if word_counts[word] >= threshold]
    print 'Filtered words from %d to %d' % (len(word_counts), len(vocab))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx_to_word = {0: u'<NULL>' , 1: u'<START>', 2: u'<END>'}
    
    idx = 4
    for word in vocab:
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        idx += 1

    return word_to_idx, idx_to_word, max_len

def build_caption_vectors(annotations, word_to_idx, max_len=15):
    '''
    Ensures that all words in train caption should be included in word_to_idx
    Inputs:
        annotations: annotations
        word_to_idx: word to index dictionary
        max_len: max length of captions
    Returns:
        captions: caption vectors
    '''
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_len+2)).astype(np.int32)   # two for special tokens: <'START'> and <'END'> 

    for i, sentence in enumerate(annotations['caption']):
        words = sentence.lower().split(' ')
        n_words = len(words)
        capvec = []
        
        capvec.append(word_to_idx['<START>'])
        for word in words:
            assert word in word_to_idx
            capvec.append(word_to_idx[word])
        capvec.append(word_to_idx['<END>'])

        if len(capvec) < (max_len + 2):
            for j in range(max_len + 2 - len(capvec)):
                capvec.append(word_to_idx['<NULL>'])
                
        captions[i, :] = np.asarray(capvec)
    print "success building caption vectors: ", captions.shape
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