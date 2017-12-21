
import hickle
import json
import tensorflow as tf

from config import *
from core.vggnet import Vgg19, load_and_resize_image
from core.utils import *
from core.solver import CaptioningSolver
from core.model import CaptionGenerator

test_image_dir = TEST_DATA_PATH + '/caption_test1_images_20170923/'
batch_size = 64
SPLIT = 'test'


def main(solver):
    image_feature_save_path = './data/%s/%s.features.hkl' % (SPLIT, SPLIT)
    file_name_save_path = './data/%s/%s.file_names.hkl' % (SPLIT, SPLIT)

    # extract image VGG feature
    if not (os.path.exists(image_feature_save_path) and os.path.exists(file_name_save_path)):
        file_names = os.listdir(test_image_dir)
        image_path = map(lambda _path: test_image_dir + _path, file_names)
        n_examples = len(image_path)

        all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)
        vggnet = Vgg19(VGG_MODEL_PATH)
        vggnet.build()
        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: load_and_resize_image(x), image_batch_file)).astype(
                    np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, SPLIT))

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, image_feature_save_path)
            hickle.dump(file_names, file_name_save_path)
            print ("Saved %s.." % (image_feature_save_path))
            print ("Saved %s.." % (file_name_save_path))
    else:
        all_feats = hickle.load(image_feature_save_path)
        file_names = hickle.load(file_name_save_path)
        print ("Loaded %s.." % (image_feature_save_path))
        print ("Loaded %s.." % (file_name_save_path))
    # generate captions dump as json
    _, _, generated_captions = solver.model.build_sampler(max_len=20)
    all_gen_cap = np.ndarray((all_feats.shape[0], 20))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, solver.test_model)

        n_iters_val = int(np.ceil(float(all_feats.shape[0])/batch_size))
        for i in range(n_iters_val):
            features_batch = all_feats[i*batch_size:(i+1)*batch_size]
            feed_dict = {solver.model.features: features_batch}
            gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
            all_gen_cap[i*batch_size:(i+1)*batch_size] = gen_cap

    all_decoded = decode_captions(all_gen_cap, solver.model.idx_to_word)

    all_decoded = map(lambda x: x.replace(' ', ''), all_decoded)

    evaluated_captions = []
    for idx, caption in enumerate(all_decoded):
        image_id = file_names[idx].split('.')[0]
        evaluated_captions.append({'image_id': image_id, 'caption': caption})
    json.dump(evaluated_captions, open('./data/%s/%s.evaluated.json' % (SPLIT, SPLIT), 'w'))


if __name__ == '__main__':
    data = load_coco_data(data_path='./data', split='val')
    with open('./data/train/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    _model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=2000, n_time_step=16, prev2out=True,
                                                 ctx2out=True, alpha_c=0.5, selector=True, dropout=True)
    solver = CaptioningSolver(_model, data, data, n_epochs=15, batch_size=128, update_rule='adam',
                                          learning_rate=0.001, print_every=500, save_every=1, image_path='/mnt/ai_challenger_caption_validation_20170910/caption_validation_images_20170910',
                                    pretrained_model=None, model_path='./model/lstm6', test_model='./model/lstm6/model-16',
                                     print_bleu=False, log_path='./log/')
    main(solver)