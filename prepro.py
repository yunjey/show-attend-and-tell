import json
import jieba
import pandas as pd
from core.utils import *
from core.vggnet import Vgg19, load_and_resize_image
from collections import Counter
import os
import sys
from scipy import ndimage
import hickle
import tensorflow as tf
from PIL import Image
from config import *
import hashlib

def fenci(sentence):
    return ' '.join(jieba.cut(sentence))


def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        raw_data = json.load(f)
    print "there are %d samples in %s" % (len(raw_data), caption_file)

    caption_data = []
    miss_count = 0
    for sample in raw_data:
        image_file_name = os.path.join(image_dir, sample['image_id'])
        if not os.path.exists(image_file_name):
            miss_count += 1
            if miss_count % 100 == 0:
                print "miss count %d " % miss_count
            continue
        for caption in sample['caption']:
            sample_dict = {'caption': caption,
                           'fenci_caption': fenci(caption),
                           'image_file_name': image_file_name,
                          'image_id': sample['image_id']}
            caption_data.append(sample_dict)

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(caption_data)
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)

    return caption_data


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['fenci_caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx

def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)

    for i, caption in enumerate(annotations['fenci_caption']):
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
            if len(cap_vec) > max_length:
                break
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions

def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['image_file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx

def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs

def generate_contest_reference(annotations, split):
    # prepare reference captions to compute bleu scores later
    ref_images = []
    ref_annotations = []
    image_ids = set()
    i = -1
    id = 1
    for fenci_caption, image_id in zip(annotations['fenci_caption'], annotations['image_id']):
        image_hash = int(int(hashlib.sha256(image_id).hexdigest(), 16) % sys.maxint)
        if not image_id in image_ids:
            image_ids.add(image_id)
            i += 1
            ref_images.append({'file_name': image_id, 'id': image_hash})
        ref_annotations.append({'caption': fenci_caption, 'id': id, 'image_id': image_hash})
        id += 1
    result = {'annotations': ref_annotations, 'images': ref_images, "type": "captions", 'info': {}, 'licenses': {}}
    print "Finished building %s caption dataset" %split
    return result

def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1

    train_caption_file = TRAIN_DATA_PATH + '/caption_train_annotations_20170902.json'
    image_dir = TRAIN_DATA_PATH + '/caption_train_images_20170902/'
    val_caption_file = VAL_DATA_PATH + '/caption_validation_annotations_20170910.json'
    val_image_dir = VAL_DATA_PATH + '/caption_validation_images_20170910/'
    # test_image_dir = TEST_DATA_PATH + '/caption_validation_images_20170910/'

    train_dataset = _process_caption_data(train_caption_file, image_dir, max_length)
    val_dataset = _process_caption_data(val_caption_file, val_image_dir, max_length)
    # test_dataset = _process_caption_data(test_caption_file, test_image_dir, max_length)
    # init make dirs
    sub_train_split = ['train' + str(i) for i in range(21)]
    # split_parts = ['train', 'val', 'test'] + sub_train_split
    split_parts = ['train', 'val'] + sub_train_split
    for split in split_parts:
        path = 'data/' + split
        if not os.path.exists(path):
            os.makedirs(path)

    save_pickle(train_dataset, 'data/train/train.annotations.pkl')
    save_pickle(val_dataset, 'data/val/val.annotations.pkl')
    # save_pickle(test_dataset, 'data/test/test.annotations.pkl')

    # since the dataset might larger than system memory, cut the train dataset into several parsts
    block_size = len(train_dataset)/21
    for i in range(21):
        save_pickle(train_dataset[i*block_size: (i+1)*block_size].reset_index(drop=True), 'data/train%d/train%d.annotations.pkl' % (i, i))

    for split in split_parts:
        annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))

        if split == 'train':
            word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            save_pickle(word_to_idx, './data/%s/word_to_idx.pkl' % split)

        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, './data/%s/%s.captions.pkl' % (split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))

        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))

        reference_json = generate_contest_reference(annotations, split)
        json.dump(reference_json, open('./data/%s/%s.references.json' % (split, split), 'w'))

    # extract conv5_3 feature vectors
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    tf.reset_default_graph()
    vggnet = Vgg19(VGG_MODEL_PATH)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for split in split_parts[1:]:
            anno_path = './data/%s/%s.annotations.pkl' % (split, split)
            save_path = './data/%s/%s.features.hkl' % (split, split)
            annotations = load_pickle(anno_path)
            image_path = list(annotations['image_file_name'].unique())
            n_examples = len(image_path)

            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: load_and_resize_image(x), image_batch_file)).astype(
                    np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, split))

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))

if __name__ == '__main__':
    main()
