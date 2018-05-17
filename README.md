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

The test accuracy：
2018-05-16 17:14:05.465694: E tensorflow/stream_executor/cuda/cuda_driver.cc:936] failed to allocate 4.00G (4294967296 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY
2018-05-16 17:14:05.465722: W tensorflow/core/common_runtime/bfc_allocator.cc:219] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.60GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
test accuracy0.9886
end time:  2018-05-16  17:14:06

Process finished with exit code 0
