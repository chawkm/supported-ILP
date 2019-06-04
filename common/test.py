import tensorflow as tf
from tensorflow.contrib import autograph


@autograph.convert()
def nearest_odd_square(x):
  if x > 0:
    x = x * x
    if x % 2 == 0:
      x = x + 1
  return x

with tf.Graph().as_default():
  with tf.Session() as sess:
    print(sess.run(nearest_odd_square(tf.constant(4))))
    print(sess.run(nearest_odd_square(tf.constant(5))))
    print(sess.run(nearest_odd_square(tf.constant(6))))


x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    y = x * x * x
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0