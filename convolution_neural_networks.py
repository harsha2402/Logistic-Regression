import tensorflow as tf
import numpy
import os
import imageio
from tensorflow.examples.tutorials.mnist import input_data
from skimage import color
from skimage.transform import resize

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def oneHotEncoding(self,t):
    print ("oneHotEncoding")
    target = numpy.zeros(len(t),10)
    for i in range(len(t)):
        index = t[i]
        target[i][int(index[0])] = 1
    return target


numpy.random.seed(0)
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
usps_in = numpy.zeros((1,784))
usps_lab = numpy.zeros((1,1))
path= "C:/Users/harsh/Desktop/UB NOTes/CSE 574/PROJECT 3/proj3_images/Test/"
val = 9
count=0
for name in os.listdir(path):
    final_path2 = path + name
    if name != "Thumbs.db":
        count = count +1
        if int(name.split('.')[0].split('_')[1])%150==0:
            val = val - 1
        img = imageio.imread(final_path2)
        gray_img = color.rgb2gray(img)
        resized_img = resize(gray_img,(28,28)) / 255.0
        flat_img = numpy.reshape(resized_img, (784,))
        usps_in = numpy.insert(usps_in,len(usps_in),flat_img,axis=0)
        usps_lab = numpy.insert(usps_lab,len(usps_lab),int(val),axis=0)
        usps_lab = oneHotEncoding(usps_lab)

inpu = tf.placeholder(tf.float32, [None, 784])
pred_out = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
weights = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))
aout = tf.nn.softmax(tf.matmul(inpu, weights) + bias)
epoch=1000
batch_size=100

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(inpu, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pred_out, logits=y_conv))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(pred_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
            batch = mnist.train.next_batch(100)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={inpu: batch[0], pred_out: batch[1], keep_prob: 1.0})
                train_step.run(feed_dict={inpu: batch[0], pred_out: batch[1], keep_prob:0.5})
    print('Total test accuracy %g' % accuracy.eval(feed_dict={inpu: mnist.test.images, pred_out: mnist.test.labels, keep_prob: 1.0}))
    print("Accurancy Usps:", accuracy.eval(feed_dict={inpu: usps_in, pred_out: usps_lab}))






