import numpy as np
import tensorflow as tf
import glob, os

# Import MNIST default_graph
from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data in /tmp/input_data
path = 'mnist_data'

print('Importing MNIST data...')
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
print('MNIST data imported.')

train_batch_size = 5000
test_batch_size = 200

x_train, y_train = mnist.train.next_batch(train_batch_size)
x_test, y_test = mnist.test.next_batch(test_batch_size)

train_digits_pl = tf.placeholder('float', [None, 784])
test_digit_pl = tf.placeholder('float', [784])

# Nearest Neighbor calculation using L1 distance
l1_distance = tf.abs(tf.add(train_digits_pl, tf.negative(test_digit_pl)))
distance = tf.reduce_sum(l1_distance, axis=1)

# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)
accuracy = 0.

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(x_test)):
        # Get nearest Neighbor
        nn_index = sess.run(pred, feed_dict={train_digits_pl: x_train, test_digit_pl: x_test[i,:]})

        predValue = np.argmax(y_train[nn_index])
        trueValue = np.argmax(y_test[i])
        # Get nearest neighbor class label and compare it to its true label
        print('Test', i, 'Prediction:', predValue, 'True label:', trueValue)

        if (predValue == trueValue):
            accuracy += 1./len(x_test)

    print('Done')
    print('Accuracy', accuracy)
