import tensorflow as tf
import numpy
import os
import imageio
from tensorflow.examples.tutorials.mnist import input_data
from skimage import color
from skimage.transform import resize


def before_hid(x):
    hidout = tf.add(tf.matmul(x, w1), b1)
    temp=tf.nn.sigmoid(hidout)
    return temp

def after_hid(x):
    output=tf.nn.softmax(tf.add(tf.matmul(x,w2),b2))
    return output

def oneHotEncoding(self,t):
    print ("oneHotEncoding")
    target = numpy.zeros(len(t),10)
    for i in range(len(t)):
        index = t[i]
        target[i][int(index[0])] = 1
    return target

numpy.random.seed(0)
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
usps_in = numpy.zeros((1,784))
usps_lab = numpy.zeros((1,1))
path2 = "C:/Users/harsh/Desktop/UB NOTes/CSE 574/PROJECT 3/proj3_images/Test/"
val = 9
count=0
for name in os.listdir(path2):
    final_path2 = path2 + name
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

training = 500
batch_size = 100
neorons = 300
no_input = 784
no_output = 10 
inputi = tf.placeholder("float", [None, no_input])
labels = tf.placeholder("float", [None, no_output])

w1 = tf.Variable(tf.random_normal([no_input, neorons]))
w2 = tf.Variable(tf.random_normal([neorons, no_output]))
b1 = tf.Variable(tf.random_normal([neorons]))
b2 = tf.Variable(tf.random_normal([no_output]))           

bhid=before_hid(inputi)
ahid=after_hid(bhid)
#now you have the final neural network output. Forward propogation ends here.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ahid, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(training):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_img, batch_lab = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={inputi: batch_img,labels: batch_lab})
            avg_cost += c / total_batch
    print("Epoch:", (i + 1), "cost =", "{:.3f}".format(avg_cost))
    correct_prediction = tf.equal(tf.argmax(ahid, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy MNIST:", accuracy.eval({inputi: mnist.test.images, labels: mnist.test.labels}))
    print("Accurancy Usps:", accuracy.eval(feed_dict={inputi: usps_in, labels: usps_lab}))