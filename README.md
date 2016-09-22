# Show, Attend and Tell 
This is a tensorflow implementation for [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)

## Reference
Author's source code: https://github.com/kelvinxu/arctic-captions

Another tensorflow implementation: https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow

(There are some bugs and visualization result is not good.)

## Result
Below are visualizations of the (soft) attention weights for each generated word.

If you want to get more train details or visualization results, see `Attention-400000-lstm.ipynb`. 

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

#####Generated Caption: A plane flying over a body of water.
![alt text](jpg/test.jpg "test image")

#####Generated Caption: A zebra standing in the grass near a tree.
![alt text](jpg/test2.jpg "test image")
