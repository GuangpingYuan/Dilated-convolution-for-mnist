# Dilated-convolution-for-mnist
Improve the mnist classification using dilated convolutions.
This code is written to try the dilated convolutional structure in mnist dataset.This code is based on the CNN Mnist classifier example
in the tensorflow webseite.
I used 6 convolution layers, which dilated rate is[1,1,2,4,1,1] and a softmax output layer.The test accuracy is 98.86%, it's better than 
the original structure.
