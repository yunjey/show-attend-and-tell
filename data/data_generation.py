import cPickle as pickle
from utils import *


def main():
    splits = ['train', 'val', 'test']
    for split in splits:
        annotations = load_pickle('./%s/%s.annotations.pkl' %(split, split))

        if split == 'train':
            word_to_idx = load_pickle('./train/word_to_idx.pkl')
            captions = build_caption_vectors(annotations, word_to_idx, 15)
            save_pickle(captions, './train/train.captions.pkl')

        file_names, id_to_idx = get_file_names(annotations)
        save_pickle(file_names, './%s/%s.file.names.pkl' %(split, split))

        if split == 'train':
            image_idxs = get_image_idxs(annotations, id_to_idx)
            save_pickle(image_idxs, './train/train.image.idxs.pkl')


if __name__ == "__main__":
   main()


    
    
    
    
    
    
    
    
    
    
    