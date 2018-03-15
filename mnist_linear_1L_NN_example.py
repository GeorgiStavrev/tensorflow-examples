import tensorflow as tf
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

# side is the size of the side of the image
side = 28

# m is the size of an example (vector size)
m = side * side

# n is the number of classes
n = 10

# Model: yhat = Wx + b
W = tf.Variable(tf.zeros([m, n]), name='W')
b = tf.Variable(tf.zeros([n]), name='b')

x = tf.placeholder(tf.float32, shape=[None, m], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10], name='y')

logits = tf.matmul(x, W) + b
yhat = tf.nn.softmax(logits)

# define loss
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

# Activation + Optimization: Softmax + GradientDescent
# lr is learning read_data_sets
lr = 0.5

optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(cross_entropy_loss)

init = tf.global_variables_initializer()

# epochs is epochs the model will be trained
epochs = 1000
print_step = 100
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        sess.run(train, feed_dict={x:x_train,y:y_train})
        if epoch % print_step == 0:
            pred = sess.run(yhat, feed_dict={x:x_test})
            accuracy = 0.

            #accuracy = sess.run(tf.metrics.accuracy(labels=tf.argmax(y_test,0), predictions=tf.argmax(pred,0)))
            correct_pred = tf.equal(tf.argmax(y_test,1), tf.argmax(pred,1))
            accuracy = sess.run(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))
            print('Accuracy at epoch ' + str(epoch) + ' is: ', str(round(accuracy*100, 2)) + '% (Error is ' + str(round(100 - accuracy * 100,2)) + '%)')

    writer = tf.summary.FileWriter('./ml_example', sess.graph)
    writer.close()
