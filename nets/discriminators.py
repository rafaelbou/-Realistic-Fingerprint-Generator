from ops import *


def discriminator(dcgan, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        h0 = lrelu(conv2d(image, dcgan.disc_input_layer_depth, s_h=1, s_w=1, name='d_h0_conv'))
        h1 = lrelu(dcgan.d_bn1(conv2d(h0, dcgan.disc_input_layer_depth * 2, name='d_h1_conv')))
        h2 = lrelu(dcgan.d_bn2(conv2d(h1, dcgan.disc_input_layer_depth * 4, name='d_h2_conv')))
        h3 = lrelu(dcgan.d_bn3(conv2d(h2, dcgan.disc_input_layer_depth * 8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [dcgan.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4


def discriminator_3_fc(dcgan, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        h0 = lrelu(conv2d(image, dcgan.disc_input_layer_depth, s_h=1, s_w=1, name='d_h0_conv'))
        h1 = lrelu(dcgan.d_bn1(conv2d(h0, dcgan.disc_input_layer_depth * 2, name='d_h1_conv')))
        h2 = lrelu(dcgan.d_bn2(conv2d(h1, dcgan.disc_input_layer_depth * 4, name='d_h2_conv')))
        h3 = lrelu(dcgan.d_bn3(conv2d(h2, dcgan.disc_input_layer_depth * 8, name='d_h3_conv')))
        h4 = lrelu(dcgan.d_bn4(conv2d(h3, dcgan.disc_input_layer_depth * 16, name='d_h4_conv')))
        h5 = lrelu(dcgan.d_bn5(conv2d(h4, dcgan.disc_input_layer_depth * 32, name='d_h5_conv')))
        h6 = lrelu(linear(tf.reshape(h5, [dcgan.batch_size, -1]), 1000, 'd_h6_lin'))
        h7 = lrelu(linear(h6, 100, 'd_h7_lin'))
        h8 = linear(h7, 1, 'd_h8_lin')

        return tf.nn.sigmoid(h8), h8


discriminators = {"basline": discriminator,
                  "discriminator_3_fc": discriminator_3_fc}


def discriminator_factory(dcgan, image, reuse=False, discriminator_name="discriminator_3_fc"):
    if discriminator_name not in discriminators.keys():
        assert ('Invalid generator name! {} is not know. Valid generators: {}'.format(discriminator_name,
                                                                                      discriminators.keys()))
    return discriminators[discriminator_name](dcgan, image, reuse=reuse)