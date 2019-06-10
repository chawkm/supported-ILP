import tensorflow as tf
from tensorflow.contrib import autograph

# tf.enable_eager_execution()

all_pairs = [(x,y) for x in range(10) for y in range(10) if x != y]

ws = tf.Variable(tf.random_uniform([10], maxval=1, dtype=tf.float32), dtype=tf.float32)

print(all_pairs)


class PairLosses(object):
    @autograph.convert()
    def pair_loss(self, x):
        s1 = x[0]
        s2 = x[1]
        if s1 < s2:
            # if ws[s2] < ws[s1] + 1.0:
            return tf.sigmoid(50*(ws[s2] - ws[s1]))
            # return tf.log(1 + tf.exp(-(ws[s1] - ws[s2])))
            # else:
            #     return 0.0
        else:
            # if ws[s1] < ws[s2] + 1.0:
            return tf.sigmoid(50*(ws[s1] - ws[s2]))
            # return tf.log(1 + tf.exp(-(ws[s2] - ws[s1])))
            # else:
            #     return 0.0


# with tf.Graph().as_default():
PLosses = PairLosses()
tf_pairs = tf.constant(all_pairs)
loss = tf.reduce_mean(tf.map_fn(PLosses.pair_loss, tf_pairs, dtype=tf.float32))
global_step = tf.train.get_or_create_global_step()
opt = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(loss, global_step=global_step)


EPOCHS = 150

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):
        # print(sess.run(tf_pairs))
        # print(sess.run([ws]))
        print(sess.run([loss, opt]))

    print(sess.run([ws]))

