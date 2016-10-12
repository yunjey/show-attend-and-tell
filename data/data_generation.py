import cPickle as pickle
from preprocess import *


# load annotations
with open('./train/train.annotations.pkl', 'rb') as f:
    train_annotations = pickle.load(f)
    
    
with open('./val/val.annotations.pkl', 'rb') as f:
    val_annotations = pickle.load(f)

    
with open('./test/test.annotations.pkl', 'rb') as f:
    test_annotations = pickle.load(f)

    
with open('./train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)
    

train_captions = build_caption_vectors(train_annotations, word_to_idx, 15)
with open('./train/train.captions.pkl', 'wb') as f:
    pickle.dump(train_captions, f, pickle.HIGHEST_PROTOCOL)
    print 'generated train.captions.pkl'
    

train_file_names, id_to_idx = get_file_names(train_annotations)
with open('./train/train.file.names.pkl', 'wb') as f:
    pickle.dump(train_file_names, f, pickle.HIGHEST_PROTOCOL)
    print 'generated train.file.names.pkl'
    
    
train_image_idxs = get_image_idxs(train_annotations, id_to_idx)
with open('./train/train.captions.pkl', 'wb') as f:
    pickle.dump(train_captions, f, pickle.HIGHEST_PROTOCOL)
    print 'generated train.captions.pkl'

    
val_file_names, _ = get_file_names(val_annotations)
with open('./val/val.file.names.pkl', 'wb') as f:
    pickle.dump(val_file_names, f, pickle.HIGHEST_PROTOCOL)
    print 'generated val.file.names.pkl'

    
test_file_names, _ = get_file_names(test_annotations)
with open('./test/test.file.names.pkl', 'wb') as f:
    pickle.dump(test_file_names, f, pickle.HIGHEST_PROTOCOL)
    print 'generated test.file.names.pkl'
    
    
    
    
    
    
    
    
    
    
    
    