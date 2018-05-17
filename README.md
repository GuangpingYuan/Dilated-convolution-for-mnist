# Dilated-convolution-for-mnist
Improve the mnist classification using dilated convolutions.
This code is written to try the dilated convolutional structure in mnist dataset.This code is based on the CNN Mnist classifier example
in the tensorflow webseite.
I used 6 convolution layers, which dilated rate is[1,1,2,4,1,1] and a softmax output layer.The test accuracy is 98.86%, it's better than 
the original structure.
To run this code, you need to download the mnist dataset and input_data first.
Download mnist dataset:
http://yann.lecun.com/exdb/mnist/
Download input_data.py : 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py
