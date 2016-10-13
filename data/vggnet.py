import scipy.io
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import tensorflow as tf

class VGGNet(object):
    """
       1. load imagenet-vgg-very-deep-19.mat file
       2. create variables using the mat file
       3. build model for feature extraction
       4. for given input images, extract return conv5_3 features
       5. if std_option is true, normalize feature with standardization
       6. return conv5_3 features
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    
    def conv2d(self, x, weight, bias, name):
        with tf.variable_scope(name):
            w = tf.get_variable('w', initializer=tf.constant(weight))
            b = tf.get_variable('b', initializer=tf.constant(bias))
            return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'), b)

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    def relu(self, x):
        return tf.nn.relu(x)
    
    def build_model(self, layer_until):
        model = scipy.io.loadmat(self.model_path)
        layers = model['layers'][0]
        
        for i, layer in enumerate(layers):
            layer_name = layer[0][0][0][0]
            layer_type = layer[0][0][1][0]
            if layer_type == 'conv':
                weight= layer[0][0][2][0][0].transpose(1, 0, 2, 3)
                bias = layer[0][0][2][0][1].reshape(-1)
                if i == 0:
                    h = self.conv2d(self.images, weight, bias, layer_name)
                else:
                    h = self.conv2d(h, weight, bias, layer_name)
            elif layer_type == 'relu':
                h = self.relu(h)
            elif layer_type == 'pool':
                h = self.max_pool(h)
                
            if layer_name == layer_until:
                return h