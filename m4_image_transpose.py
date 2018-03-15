import tensorflow as tf
import matplotlib.image as mp_img
import matplotlib.pyplot as plot
import os

filename = './DandelionFlower.jpg'

image = mp_img.imread(filename)

print('Image shape: ', image.shape)

#plot.imshow(image)
#plot.show()

x = tf.Variable(image, name='x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    transpose = tf.transpose(x, perm=[1,0,2])

    result = sess.run(transpose)
    plot.imshow(result)
    plot.show()
