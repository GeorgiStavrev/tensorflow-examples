import tensorflow as tf

# y = Wx + b

W = tf.Variable([2.5, 4.0], tf.float32, name='var_W')

x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([5.0, 10.0], tf.float32, name='var_b')

y = W * x + b


# Initialize all variable defined

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print('Final result: Wx + b = ', sess.run(y, feed_dict={x:[10,100]}))


# s = Wx
s = W * x

# Initialize only those variables that you might Intermediate
init = tf.variables_initializer([W])

with tf.Session() as sess:
    sess.run(init)

    #print('Will this work?: Wx + b', sess.run(y, feed_dict={x:[10,100]}))

number = tf.Variable(2)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()

result = number.assign(tf.multiply(number, multiplier))

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        print('Result: ',sess.run(result))
        sess.run(multiplier.assign_add(1))
