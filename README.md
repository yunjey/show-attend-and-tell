# Show, Attend and Tell 
This is a TensorFlow implementation for [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)

<br/>

## References
Author's source code: https://github.com/kelvinxu/arctic-captions

Another tensorflow implementation: https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow


## Requirements
* Python 2.7
* TensorFlow
* cPickle
* hickle

## Usage 

    $ git clone https://github.com/yunjey/show-attend-and-tell.git 
    
Download MSCOCO 2014 training images and 2014 val images (you should also resize them to 224x224)

    http://mscoco.org/dataset/#download 
   
Download VGGNet imagenet-vgg-verydeep-19.mat and save into prep directory
    
To generate data for training and testing, run codes below:

    $ python prepare_data.py
    $ python prepare_features.py imagenet-vgg-verydeep-19.mat conv5_3 100

<br/>
## Result
Below are visualizations of the (soft) attention weights for each generated word.

If you want to get more training details or visualization results, see `evaluate_model.ipynb`. 
<br/>
###Training Data

#####(1) Generated Caption: A plane flying in the sky with a landing gear down.
![alt text](jpg/train2.jpg "train image")

#####(2) Generated Caption: A giraffe and two zebra standing in the field.
![alt text](jpg/train.jpg "train image")

###Validation data

#####(1) Generated Caption: A large elephant standing in a dry grass field.
![alt text](jpg/val.jpg "val image")

#####(2) Generated Caption: A baby elephant standing on top of a dirt field.
![alt text](jpg/val2.jpg "val image")

###Test data

#####(1) Generated Caption: A plane flying over a body of water.
![alt text](jpg/test.jpg "test image")

#####(2) Generated Caption: A zebra standing in the grass near a tree.
![alt text](jpg/test2.jpg "test image")


