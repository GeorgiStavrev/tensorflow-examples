import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print('Importing MNIST data...')
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
print('MNIST data imported.')

sess = tf.InteractiveSession()

m = 784
n = 10

x = tf.placeholder(tf.float32, shape=[None, m])
y = tf.placeholder(tf.float32, shape=[None, n])

# change the MNIST input data from a list of values to a 28x28 pixel x ! grayscale cube
# which the convolution can use

x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

# Define helper function to create weights and biases variables, and convolution, and pooling layers
# We are using RELU as activation function. These must be initialized to small positive numbers
# and with some noise so you don't end up going to zero when comparing diffs

def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Define layers in the NN
n_channels = 1
# 1st Conv layers
# 32 features for each 5x5 patch of the images
n_filters_1 = 32

with tf.name_scope('Conv2D_1'):
    W_conv1 = weight_var([5, 5, n_channels, n_filters_1]) # 32 filters, 1 channel, 5x5 filter
    b_conv1 = bias_var([n_filters_1]) #

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# 2nd Conv Layer + MaxPool
n_filters_2 = 64
with tf.name_scope('Conv2D_2'):
    W_conv2 = weight_var([5, 5, n_filters_1, n_filters_2])
    b_conv2 = bias_var([n_filters_2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# Flatten
with tf.name_scope('Flatten'):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*n_filters_2])

# Fully connecter layers
n_fc_neurons = 1024
with tf.name_scope('FC_1'):
    W_fc1 = weight_var([7*7*n_filters_2, n_fc_neurons])
    b_fc1 = bias_var([n_fc_neurons])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layers
with tf.name_scope('FC_2_Output'):
    W_fc2 = weight_var([n_fc_neurons, n])
    b_fc2 = bias_var([n])

    # Define Model
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss measurement
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))

# optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# How accuracte is it?
correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

writer = tf.summary.FileWriter('./mnist_cnn', sess.graph)
sess.run(tf.global_variables_initializer())

# Train the model
import time

num_steps = 3000
display_every = 100
batch_size = 500

start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
        end_time = time.time()
        print('step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%'.format(i, end_time-start_time, train_accuracy*100))

end_time = time.time()
print('Total training time for {0} batches: {1:.2f} seconds'.format(i+1, end_time - start_time))

print('Test accuracy {0:.3f}%'.format(accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})*100))

writer.close()
sess.close()
