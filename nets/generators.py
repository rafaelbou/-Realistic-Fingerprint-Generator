from ops import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class generator:
    def __init__(self, dcgan, z, maps=None, reuse=False):
        with tf.variable_scope("generator") as scope:
            self.maps = maps
            self.reuse = reuse
            if reuse:
                scope.reuse_variables()
            if maps is not None:
                self.G_encoder = self.generator_encoder(dcgan)
                self.noisy_nimute_vec = tf.add(self.G_encoder, z)
            else:
                self.noisy_nimute_vec = z
            self.G_decoder = self.generator_decoder(dcgan)

    def generator_encoder(self, dcgan):
        with tf.variable_scope("generator_encoder") as scope:
            if self.reuse:
                scope.reuse_variables()
            self.e_h0 = lrelu(conv2d(self.maps, dcgan.disc_input_layer_depth, name='g_ench0_conv'))
            self.e_h1 = lrelu(dcgan.d_bn1(conv2d(self.e_h0, dcgan.disc_input_layer_depth * 2, name='g_ench_1_conv')))
            self.e_h2 = lrelu(dcgan.d_bn2(conv2d(self.e_h1, dcgan.disc_input_layer_depth * 4, name='g_ench_2_conv')))
            self.e_h3 = lrelu(dcgan.d_bn3(conv2d(self.e_h2, dcgan.disc_input_layer_depth * 8, name='g_ench_3_conv')))
            self.e_h4 = lrelu(dcgan.d_bn4(conv2d(self.e_h3, dcgan.disc_input_layer_depth * 16, name='g_ench_4_conv')))
            self.e_h5 = lrelu(dcgan.d_bn5(conv2d(self.e_h4, dcgan.disc_input_layer_depth * 32, name='g_ench_5_conv')))
            self.e_h6 = lrelu(linear(tf.reshape(self.e_h5, [dcgan.batch_size, -1]), 1000, 'g_ench_6_lin'))
            self.e_h7 = lrelu(linear(self.e_h6, 100, 'g_ench_7_lin'))

            return self.e_h7

    def generator_decoder(self, dcgan):
        with tf.variable_scope("generator_decoder") as scope:
            if self.reuse:
                scope.reuse_variables()
            s_h, s_w = dcgan.output_height, dcgan.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)

            # project `z` and reshape
            self.z_1, _, _ = linear(
                self.noisy_nimute_vec, dcgan.gen_input_layer_depth * 4 * s_h64 * s_w64, 'g_h0_lin', with_w=True)

            self.z_, _, _ = linear(
                tf.nn.relu(dcgan.g_bnz(self.z_1)), dcgan.gen_input_layer_depth * 32 * s_h64 * s_w64, 'g_h1_lin', with_w=True)

            self.d_h0 = tf.reshape(
                self.z_, [-1, s_h64, s_w64, dcgan.gen_input_layer_depth * 32])
            self.d_h0 = tf.nn.relu(dcgan.g_bn0(self.d_h0))

            self.d_h1, _, _ = deconv2d(
                self.d_h0, [dcgan.batch_size, s_h32, s_w32, dcgan.gen_input_layer_depth * 16], name='g_h1', with_w=True)
            self.d_h1 = tf.nn.relu(dcgan.g_bn1(self.d_h1))

            self.d_h2, _, _ = deconv2d(
                self.d_h1, [dcgan.batch_size, s_h16, s_w16, dcgan.gen_input_layer_depth * 8], name='g_h2', with_w=True)
            self.d_h2 = tf.nn.relu(dcgan.g_bn2(self.d_h2))

            self.d_h3, _, _ = deconv2d(
                self.d_h2, [dcgan.batch_size, s_h8, s_w8, dcgan.gen_input_layer_depth * 4], name='g_h3', with_w=True)
            self.d_h3 = tf.nn.relu(dcgan.g_bn3(self.d_h3))

            self.d_h4, _, _ = deconv2d(
                self.d_h3, [dcgan.batch_size, s_h4, s_w4, dcgan.gen_input_layer_depth * 2], name='g_h4', with_w=True)
            self.d_h4 = tf.nn.relu(dcgan.g_bn4(self.d_h4))

            self.d_h5, _, _ = deconv2d(
                self.d_h4, [dcgan.batch_size, s_h2, s_w2, dcgan.gen_input_layer_depth * 1], name='g_h5', with_w=True)
            self.d_h5 = tf.nn.relu(dcgan.g_bn5(self.d_h5))

            if dcgan.use_maps:
                self.d_h6, _, _ = deconv2d(
                    self.d_h5, [dcgan.batch_size, s_h, s_w, int(dcgan.gen_input_layer_depth * 0.5)], name='g_h6', with_w=True)
                self.d_h6 = tf.nn.relu(self.d_h6)
                self.d_h6 = tf.concat([self.d_h6, self.maps], axis=-1)

                self.d_h7 = conv2d(self.d_h6, dcgan.c_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h7')
                return tf.nn.tanh(self.d_h7)

            self.d_h6, _, _ = deconv2d(
                self.d_h5, [dcgan.batch_size, s_h, s_w, dcgan.c_dim], name='g_h6', with_w=True)

            return tf.nn.tanh(self.d_h6)


class Unet(generator):
    def __init__(self, dcgan, z, maps=None, reuse=False):
        super(Unet, self).__init__(dcgan, z, maps=maps, reuse=reuse)
        self.G_decoder = self.generator_decoder(dcgan)

    def generator_decoder(self, dcgan):
        with tf.variable_scope("generator_decoder") as scope:
            if self.reuse:
                scope.reuse_variables()
            s_h, s_w = dcgan.output_height, dcgan.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)

            # project `z` and reshape
            self.z_1, _, _ = linear(
                self.noisy_nimute_vec, dcgan.gen_input_layer_depth * 4 * s_h64 * s_w64, 'g_h0_lin', with_w=True)

            self.z_, _, _ = linear(
                tf.nn.relu(dcgan.g_bnz(self.z_1)), dcgan.gen_input_layer_depth * 32 * s_h64 * s_w64, 'g_h1_lin', with_w=True)

            self.d_h0 = tf.reshape(
                self.z_, [-1, s_h64, s_w64, dcgan.gen_input_layer_depth * 32])
            self.d_h0 = tf.nn.relu(dcgan.g_bn0(self.d_h0))

            self.d_h1, _, _ = deconv2d(
                tf.concat([self.e_h5, self.d_h0], -1), [dcgan.batch_size, s_h32, s_w32, dcgan.gen_input_layer_depth * 16], name='g_h1', with_w=True)
            self.d_h1 = tf.nn.relu(dcgan.g_bn1(self.d_h1))

            self.d_h2, _, _ = deconv2d(
                tf.concat([self.e_h4, self.d_h1], -1), [dcgan.batch_size, s_h16, s_w16, dcgan.gen_input_layer_depth * 8], name='g_h2', with_w=True)
            self.d_h2 = tf.nn.relu(dcgan.g_bn2(self.d_h2))

            self.d_h3, _, _ = deconv2d(
                tf.concat([self.e_h3, self.d_h2], -1), [dcgan.batch_size, s_h8, s_w8, dcgan.gen_input_layer_depth * 4], name='g_h3', with_w=True)
            self.d_h3 = tf.nn.relu(dcgan.g_bn3(self.d_h3))

            self.d_h4, _, _ = deconv2d(
                tf.concat([self.e_h2, self.d_h3], -1), [dcgan.batch_size, s_h4, s_w4, dcgan.gen_input_layer_depth * 2], name='g_h4', with_w=True)
            self.d_h4 = tf.nn.relu(dcgan.g_bn4(self.d_h4))

            self.d_h5, _, _ = deconv2d(
                tf.concat([self.e_h1, self.d_h4], -1), [dcgan.batch_size, s_h2, s_w2, dcgan.gen_input_layer_depth * 1], name='g_h5', with_w=True)
            self.d_h5 = tf.nn.relu(dcgan.g_bn5(self.d_h5))

            self.d_h6, _, _ = deconv2d(
                tf.concat([self.e_h0, self.d_h5], -1), [dcgan.batch_size, s_h, s_w, int(dcgan.gen_input_layer_depth * 0.5)], name='g_h6', with_w=True)
            self.d_h6 = tf.nn.relu(self.d_h6)
            self.d_h6 = tf.concat([self.d_h6, self.maps], axis=-1)

            self.d_h7 = conv2d(self.d_h6, dcgan.c_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h7')
            return tf.nn.tanh(self.d_h7)


generators = {"decoder": generator.generator_decoder,
              "encoder_decoder": generator,
              "unet": Unet}


def generator_factory(dcgan, z, maps=None, reuse=False, generator_name="encoder_decoder"):
    if generator_name not in generators.keys():
        assert ('Invalid generator name! {} is not know. Valid generators: {}'.format(generator_name, generators.keys()))
    net = generators[generator_name](dcgan, z, maps, reuse=reuse)
    return net.G_decoder
