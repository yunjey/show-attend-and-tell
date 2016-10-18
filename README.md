# Show, Attend and Tell 
This is a TensorFlow implementation for [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)

Referenced author's theano source code: https://github.com/kelvinxu/arctic-captions


<br/>

## Usage
Optimizing the code is on progress to provide better performance and readability.

Usages for data and model will be updated soon.

<br/>

## Results
Below is a visualization of the (soft) attention weights for each generated word.

If you want to get more training details or visualization results, see `evaluate_model.ipynb`. 
<br/>
####Training data

#####(1) Generated caption: A plane flying in the sky with a landing gear down.
![alt text](jpg/train2.jpg "train image")

#####(2) Generated caption: A giraffe and two zebra standing in the field.
![alt text](jpg/train.jpg "train image")

####Validation data

#####(1) Generated caption: A large elephant standing in a dry grass field.
![alt text](jpg/val.jpg "val image")

#####(2) Generated caption: A baby elephant standing on top of a dirt field.
![alt text](jpg/val2.jpg "val image")

####Test data

#####(1) Generated caption: A plane flying over a body of water.
![alt text](jpg/test.jpg "test image")

#####(2) Generated caption: A zebra standing in the grass near a tree.
![alt text](jpg/test2.jpg "test image")


