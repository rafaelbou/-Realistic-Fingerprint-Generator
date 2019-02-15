import tensorflow as tf
import numpy as np


def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)


def l2_loss(generated, real, weights=1.):
    return tf.losses.mean_squared_error(real, generated, weights)


def l2_loss_weighted(generated, real, weights=1.):
    return tf.losses.mean_squared_error(tf.multiply(real, weights), tf.multiply(generated, weights))


def unittest():
    generated = tf.placeholder(tf.float64, [None,10,10])
    real = tf.placeholder(tf.float64, [None,10,10])
    weights = tf.placeholder(tf.float64, [None, 10,10])

    l2 = l2_loss(generated, real, weights)
    l2_w = l2_loss_weighted(generated, real, weights)
    generated_val = [np.zeros((10,10)), np.zeros((10,10))]
    real_val = [np.ones((10,10)), np.ones((10,10))]
    # real_val[:, 5:7, 5:7] = 0
    weights_val = [np.ones((10,10)), np.ones((10,10))]
    weights_val[1][5:7, 5:7] = 0
    with tf.Session() as sess:
        loss = sess.run(l2_w, feed_dict={generated: generated_val, real: real_val, weights: weights_val})
        print(loss)

# unittest()