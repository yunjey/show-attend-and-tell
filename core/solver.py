import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import math
import os 
import cPickle as pickle
from scipy import ndimage
from utils import decode_captions, sample_coco_minibatch
from homogeneous import Homogeneous_Data


class CaptioningSolver(object):
    """
    CaptioningSolver produces functions:
        - train: given images and captions, trains model / prints loss
        - test: given images, generates(or samples) captions and visualizes attention weights.
        Example usage might look something like this:
        data = load_coco_data()
        model = CaptionGenerator(word_to_idx, batch_size= 100, dim_feature=[196, 512], dim_embed=128,
                                   dim_hidden=128, n_time_step=16, cell_type='lstm', dtype=tf.float32)
        solver = CaptioningSolver(model, data, n_epochs=10, batch_size=100, update_rule='adam', 
                                                learning_rate=0.03, print_every=10, save_every=10)
        solver.train()
        solver.test()
    """
    def __init__(self, model, data, **kwargs):
        """
        Required Arguments:
        - model: a caption generator model with following functions:
            - build_model: receives features and captions then build graph where root nodes are loss and logits.  
            - build_sampler: receives features and build graph where root nodes are captions and alpha weights.
        - data: dictionary with the following keys:
            - features: feature vectors of shape (82783, 196, 512)
            - file_names: image file names of shape (82783, )
            - captions: captions of shape (400131, 17) 
            - image_idxs: indices for mapping caption to image of shape (400131, ) 
            - annotations: pandas annotation data 
            - word_to_idx: word to index dictionary
        Optional Arguments:
        - n_epochs: the number of epochs to run for during training.
        - batch_size: mini batch size.
        - update_rule: a string giving the name of an update rule among the followings: 
            - 'adam'
            - 'rmsprop'
            - 'adadelta'
            - 'adagrad'
        - learning_rate: learning rate; default value is 0.03.
        - print_every: Integer; training losses will be printed every print_every iterations.
        - save_every: Integer; model variables will be saved every save_every iterations.
        - model_path: String; path for saving model
        - test_model: String; model path for testing 
        - image_path: String; path for images (for attention visualization)
        """
        self.model = model
        self.data = data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.03)
        self.print_every = kwargs.pop('print_every', 10)
        self.save_every = kwargs.pop('save_every', 100)
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.model_path = kwargs.pop('model_path', './model/')
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self.test_batch_size = kwargs.pop('test_batch_size', 100)
        self.candidate_caption_path = kwargs.pop('candidate_caption_path', './data/')
        self.image_path = kwargs.pop('image_path', './data/train2014_resized/')
        self.test_image_path = kwargs.pop('test_image_path', './data/val2014_resized')

        # Book-keeping variables 
        self.best_model = None
        self.loss_history = []
        #self.best_val_acc = 0
        #self.train_acc_history = []
        #self.val_acc_history = []

        # Set optimizer by update rule
        if self.update_rule == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer
        elif self.update_rule == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer
        else:
            self.optimizer = tf.train.RMSPropOptimizer    # don't use. cause not implement error.



    def train(self):
        """
        Train model and print out some useful information(loss, generated captions) for debugging.  
        """
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))

        # get data
        features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']

        # random shuffle caption data
        np.random.seed(1234)
        rand_idxs = np.random.permutation(n_examples)
        captions = captions[rand_idxs]
        image_idxs = image_idxs[rand_idxs]

        # initialize class for generating same length of captions
        train_iter = Homogeneous_Data(captions, batch_size=self.batch_size)

        # build graph for training
        loss = self.model.build_model()
        _, generated_captions = self.model.build_sampler()
        optimizer = self.optimizer(self.learning_rate).minimize(loss)

        print "The number of epoch: %d" %self.n_epochs
        print "Data size: %d" %n_examples
        print "Batch size: %d" %self.batch_size
        print "Iterations per epoch: %d" %n_iters_per_epoch
        
        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.6
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver(max_to_keep=10)
            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = 1000000000
            curr_loss = 0
            for e in range(self.n_epochs):
                for i in range(n_iters_per_epoch):
                    # get batch data (lengths of all mini-batch captions are same)
                    same_caption_idxs = train_iter.get_next()
                    captions_batch = captions[same_caption_idxs]
                    image_idxs_batch = image_idxs[same_caption_idxs]
                    features_batch = features[image_idxs_batch]

                    # print initial loss
                    if e == 0 and i == 0:
                        feed_dict = { self.model.features: features_batch, self.model.captions: captions_batch }
                        gen_caps, l = sess.run([generated_captions, loss], feed_dict)
                        self.loss_history.append(l)
                        print "\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
                        print "Initial train loss: %.5f" %l
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        for j in range(1):
                            print "Generated caption: %s" %decoded[j]
                        print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n"

                    # optimize
                    feed_dict = { self.model.features: features_batch, self.model.captions: captions_batch }
                    _, l = sess.run([optimizer, loss], feed_dict)
                    curr_loss += l

                    # print info
                    if (i+1) % self.print_every == 0:
                        # print out train loss 
                        print "\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
                        print "Train loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
                        # print out ground truth
                        for j in range(1):
                            ground_truths = captions[image_idxs == image_idxs_batch[j]]
                            decoded = decode_captions(ground_truths, self.model.idx_to_word)
                            for i, gt in enumerate(decoded):
                                print "Ground truth %d: %s" %(i+1, gt)        
                        # print out generated captions                       
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        for j in range(1):
                            print "Generated caption: %s" %decoded[j]
                        print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n"
                        # save loss history
                        l = sess.run(loss, feed_dict)
                        self.loss_history.append(l)


                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss", curr_loss
                if prev_loss < curr_loss:
                    #saver.restore(sess, os.path.join(self.model_path, self.best_model))
                    self.learning_rate /= 2
                    print "Reduce learning rate to %f..!" %self.learning_rate
                else:
                    prev_loss = curr_loss
                    # save best model
                    if (e+1) % self.save_every == 0:
                        saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                        print "model-%s saved." %(e+1)
                        self.best_model = 'model-%s' %(e+1)
                curr_loss = 0
                print "Best model: %s" %self.best_model
        
                
    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Sample captions and visualize attention weights for image data
        Save sampled captions in pickle file

        Inputs:
        - data: dictionary with the following keys:
            - features: feature vectors of shape (5000, 196, 512)
            - file_names: image file names of shape (5000, )
            - captions: captions of shape (24210, 17) 
            - image_idxs: indices for mapping caption to image of shape (24210, ) 
            - features_to_captions: mapping feature to captions (5000, 4~5)
        '''
        # get data
        features = data['features']

            
        # build graph for sampling
        max_len = 20
        alphas, sampled_captions = self.model.build_sampler(max_len)    # (N, max_len, L), (N, max_len)
        
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            # restore trained model
            saver = tf.train.Saver(max_to_keep=10)
            saver.restore(sess, self.test_model)
            
            # actual test step: sample captions and visualize attention
            _, features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = { self.model.features: features_batch }
            alps, sam_cap = sess.run([alphas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            if attention_visualization:
                # visualize 10 images and captions 
                for n in range(10):
                    print "Sampled Caption: %s" %decoded[n]

                    # plot original image
                    if split is 'train':
                        image_path = self.image_path
                    else:
                        image_path = self.test_image_path
                    img_path = os.path.join(image_path, image_files[n])
                    img = ndimage.imread(img_path)
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # plot image with attention weights
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t>18:
                            break
                        plt.subplot(4, 5, t+2)
                        plt.text(0, 1, words[t], color='black', backgroundcolor='white', fontsize=12)
                        plt.imshow(img)
                        alp_curr = alps[n,t,:].reshape(14,14)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.8)
                        plt.axis('off')
                    plt.show()

            if save_sampled_captions:
                # sample captions for all dataset and save into pickle format
                all_sam_cap = np.ndarray((features.shape[0], max_len))
                num_iter = int(np.ceil(float(features.shape[0]) / self.test_batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.test_batch_size:(i+1)*self.test_batch_size]
                    feed_dict = { self.model.features: features_batch }
                    all_sam_cap[i*self.test_batch_size:(i+1)*self.test_batch_size] = sess.run(sampled_captions, feed_dict)  

                # decode all sampled captions
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                with open(os.path.join(self.candidate_caption_path, "%s/%s.candidate.captions.pkl" %(split,split)), 'wb') as f:
                    pickle.dump(all_decoded, f, pickle.HIGHEST_PROTOCOL)
                    print "saved %s.candidate.captions.pkl.." %split




  