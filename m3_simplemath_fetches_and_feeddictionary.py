import tensorflow as tf

# y = Wx + b

W = tf.constant([10, 100], name='const_W')

# Note that tehse placeholders can hold tensors of any shape
x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

Wx = tf.multiply(W, x, name='Wx')
y = tf.add(Wx, b, name='y')
y_ = tf.subtract(x, b, name='y_')

with tf.Session() as sess:
    print('Wx: ', sess.run(Wx, feed_dict={x:[1,2]}))
    print('y: ', sess.run(y, feed_dict={x:[1,2], b:3}))
    print('Intermediate Wx for y: ', sess.run(y, feed_dict={Wx:[100,200], b:[3,5]}))
    print('Two results: [y, y_]: ', sess.run(fetches=[y,y_], feed_dict={x:[5,50], b:2}))

writer = tf.summary.FileWriter('./m3_example2', sess.graph)
writer.close()
