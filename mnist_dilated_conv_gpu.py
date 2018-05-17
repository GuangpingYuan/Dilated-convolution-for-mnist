import os
#choose GPU
os.environ["CUDA VISIBLE_DEVICES"]="0"

#import data
import input_data
mnist= input_data.read_data_sets("MNIST_data/",one_hot=True)

import time
start_time = time.time()
print('start time: ',time.strftime("%Y-%m-%d  %H:%M:%S",time.localtime()))

import tensorflow as tf
x = tf.placeholder("float",[None,784])
y_ = tf.placeholder("float",[None,10])

#define weight and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


#first convolutional layer,kernal size 3*3,dilation rate=1, 1 input channel,32 output channel
W_conv1 = weight_variable([3,3,1,32])
b_conv1 = bias_variable([32])

#[batch,weidth,height,number of channel(grey)]
x_image =tf.reshape(x,[-1,28,28,1])

c_dilat1 =tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1],padding='SAME',use_cudnn_on_gpu=True,
                       data_format='NHWC', dilations=[1, 1, 1, 1], name='dilation1')
h_conv1 = tf.nn.relu(c_dilat1+b_conv1)


#second convolutional layer
W_conv2 = weight_variable([3,3,32,32])
b_conv2 = bias_variable([32])
c_dilat2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1,1,1,1], padding= 'SAME',use_cudnn_on_gpu=True,
                        data_format='NHWC', dilations=[1, 1, 1, 1], name= 'dilation2')
h_conv2 = tf.nn.relu(c_dilat2+b_conv2)

#third convolutional layer
W_conv3 = weight_variable([3,3,32,32])
b_conv3 = bias_variable([32])
c_dilat3 = tf.nn.conv2d(h_conv2, W_conv3, strides= [1,1,1,1], padding= 'SAME',use_cudnn_on_gpu=True,
                        data_format= 'NHWC', dilations= [1, 2, 2, 1], name= 'dilation3')
h_conv3 = tf.nn.relu(c_dilat3 + b_conv3)

#fourth convolutional layer
W_conv4 = weight_variable([3,3,32,32])
b_conv4 = bias_variable([32])
c_dilat4 = tf.nn.conv2d(h_conv3, W_conv4, strides= [1,1,1,1], padding= 'SAME',use_cudnn_on_gpu=True,
                        data_format= 'NHWC', dilations= [1, 4, 4, 1], name='dilation4')
h_conv4 = tf.nn.relu(c_dilat4 + b_conv4)

#fifth layer
W_conv5 = weight_variable([3,3,32,32])
b_conv5 = bias_variable([32])
c_dilat5 = tf.nn.conv2d(h_conv4, W_conv5, strides= [1,1,1,1], padding= 'SAME',use_cudnn_on_gpu=True,
                        data_format= 'NHWC', dilations= [1, 1, 1, 1], name='dilation5')
h_conv5 = tf.nn.relu(c_dilat5 + b_conv5)

#1*1 layer
W_conv6 = weight_variable([1,1,32,10])
b_conv6 = bias_variable([10])
c_dilat6 = tf.nn.conv2d(h_conv5, W_conv6,strides= [1,1,1,1], padding= 'SAME',use_cudnn_on_gpu=True,
                        data_format= 'NHWC', dilations= [1,1,1,1], name= 'dilation6')
c_flat = tf.reshape(c_dilat6,[-1,28*28*10])

import time
run_time = time.time()
print(' time: ',time.strftime("%Y-%m-%d  %H:%M:%S",time.localtime()))
#output layer
W_fc2 = weight_variable([28*28*10,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(c_flat, W_fc2)+b_fc2)

#evaluation
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

#save the model
#saver = tf.train.Saver()

#run
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0],y_:batch[1]})
        print("step %d,training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0],y_: batch[1]})

print("test accuracy%g" % accuracy.eval(feed_dict={x:mnist.test.images,
                                                   y_:mnist.test.labels}))
#save model
#saver.save(sess,"model.ckpt")
#show training time
end_time = time.time()
print('end time: ',time.strftime("%Y-%m-%d  %H:%M:%S",time.localtime()))

