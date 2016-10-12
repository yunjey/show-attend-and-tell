import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import cPickle as pickle
from scipy import ndimage
from utils import decode_captions, sample_coco_minibatch
from bleu import evaluate


class CaptioningSolver(object):
    """
    CaptioningSolver produces functions:
        - train: given images and captions, trains model / prints loss
        - test: given images, generates(or samples) captions and visualizes attention weights.
        Example usage might look something like this:
        data = load_coco_data(data_path='./data', split='train', feature='conv5_3')
        word_to_idx = data['word_to_idx']
        model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                   dim_hidden=2048, n_time_step=16, cell_type='lstm', prev2out=True, 
                                             ctx2out=True, alpha_c=1.0, selector=True, use_dropout=True)
        solver = CaptioningSolver(model, data, n_epochs=30, batch_size=128, update_rule='adam',
                                      learning_rate=0.001, print_every=3000, save_every=2, image_path='./image/train2014_resized',
                                pretrained_model=None, model_path='./model/lstm', test_model='./model/lstm/model-20', test_batch_size=100,
                                 candidate_caption_path='./data/', test_image_path='./image/val2014_resized')
        solver.train()
        solver.test()
    """
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
        - model: a caption generator model with following functions:
            - build_model: receives features and captions then build graph where root node is loss.  
            - build_sampler: receives features and build graph where root nodes are captions and alpha(attention) weights.
        - data: training data; dictionary with the following keys:
            - features: feature vectors of shape (82783, 196, 512)
            - file_names: image file names of shape (82783, )
            - captions: captions of shape (400131, 17) 
            - image_idxs: indices for mapping caption to image of shape (400131, ) 
            - word_to_idx: mapping dictionary from word to index 
        - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
        - n_epochs: the number of epochs to run for during training.
        - batch_size: mini batch size.
        - update_rule: a string giving the name of an update rule among the followings: 
            - 'sgd'
            - 'momentum'
            - 'adam'
            - 'adadelta'
            - 'adagrad'
            - 'rmsprop' (don't use this. it will cause "not implement error".)
        - learning_rate: learning rate; default value is 0.03.
        - print_every: Integer; training losses will be printed every print_every iterations.
        - save_every: Integer; model variables will be saved every save_every epoch.
        - image_path: String; train image path (for attention visualization)
        - test_image_path: String; val/test image path (for attention visualization)
        - pretrained_model: String; pretrained model path 
        - model_path: String; model path for saving 
        - test_model: String; model path for test 
        - test_batch_size: Integer; batch size for test step
        - candidate_caption_path: String; path for saving sampled captions for given images 
        """
        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.03)
        self.print_every = kwargs.pop('print_every', 10)
        self.save_every = kwargs.pop('save_every', 100)
        self.image_path = kwargs.pop('image_path', './data/train2014_resized/')
        self.test_image_path = kwargs.pop('test_image_path', './data/val2014_resized')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.model_path = kwargs.pop('model_path', './model/')
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self.test_batch_size = kwargs.pop('test_batch_size', 100)
        self.candidate_caption_path = kwargs.pop('candidate_caption_path', './data/')
        self.print_bleu = kwargs.pop('print_bleu', True)

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
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer    # don't use this. it will cause "not implement error".



    def train(self):
        """
        For given feature vectors and ground truth captions, solver trains model using gpu. 
        solver also prints out some useful information such as loss and sampled caption.
        For each epoch, model compares previous loss and current loss. 
        If current loss is bigger than previous, then learning rate is decreased to half.
        """
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))

        # training dataset
        features = self.data['features']
        captions = self.data['captions']
        
        # validation dataset
        val_features = self.val_data['features']
        n_iters_val = int(np.ceil(float(val_features.shape[0]) / self.test_batch_size))

        # build graph for training
        loss = self.model.build_model()
        max_len = 20
        _, generated_captions = self.model.build_sampler(max_len=max_len)
        optimizer = self.optimizer(self.learning_rate).minimize(loss)

        print "The number of epoch: %d" %self.n_epochs
        print "Data size: %d" %n_examples
        print "Batch size: %d" %self.batch_size
        print "Iterations per epoch: %d" %n_iters_per_epoch
        
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.per_process_gpu_memory_fraction=0.9
        #config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver(max_to_keep=40)
            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = 1000000000
            curr_loss = 0
            start_t = time.time()
            for e in range(self.n_epochs):
                # random shuffle caption data
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                features = features[rand_idxs]
                for i in range(n_iters_per_epoch):
                    # get batch data (lengths of all mini-batch captions are same)
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]

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
                        for j in range(2):
                            ground_truth = captions_batch[j]
                            decoded = decode_captions(ground_truth, self.model.idx_to_word)
                            print "Ground truth %d: %s" %(j+1, decoded[0])        
                        # print out generated captions                       
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        for j in range(2):
                            print "Generated caption: %s" %decoded[j]
                        print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n"
                        # save loss history
                        l = sess.run(loss, feed_dict)
                        self.loss_history.append(l)
    
                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                
                # print out BLEU scores
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], max_len))
                    for i in range(n_iters_val):
                        features_batch = val_features[i * self.test_batch_size:(i + 1) * self.test_batch_size]
                        feed_dict = {self.model.features: features_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)  # (N, max_len)
                        all_gen_cap[i * self.test_batch_size:(i + 1) * self.test_batch_size] = gen_cap

                    # decode all sampled captions
                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    with open(os.path.join(self.candidate_caption_path, "val/val.candidate.captions.pkl"), 'wb') as f:
                        pickle.dump(all_decoded, f, pickle.HIGHEST_PROTOCOL)
                        print "saved val.candidate.captions.pkl.."
                    
                    # get bleu scores
                    final_scores = evaluate(data_path='./data', split='val', get_scores=True)
                    
                    # file write
                    if e == 0:
                        file_mode = 'w'
                    else:
                        file_mode = 'a'
                    with open(os.path.join(self.model_path, 'val.bleu.scores.txt'), file_mode) as f:
                        f.write('Epoch %d\n' %(e+1))
                        f.write('Bleu_1: %f\n' %final_scores['Bleu_1'])
                        f.write('Bleu_2: %f\n' %final_scores['Bleu_2'])
                        f.write('Bleu_3: %f\n' %final_scores['Bleu_3'])  
                        f.write('Bleu_4: %f\n' %final_scores['Bleu_4']) 
                        f.write('METEOR: %f\n' %final_scores['METEOR'])  
                        f.write('ROUGE_L: %f\n' %final_scores['ROUGE_L'])  
                        f.write('CIDEr: %f\n\n' %final_scores['CIDEr'])
                
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
        config.gpu_options.allow_growth = True
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