from ops import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Generator:
    def __init__(self, dcgan, z, maps=None, reuse=False, train=True, use_encoder=True):
        with tf.variable_scope("generator") as scope:
            self.maps = maps
            self.reuse = reuse
            self.train = train
            if reuse:
                scope.reuse_variables()
            if maps is not None and use_encoder:
                self.G_encoder = self.generator_encoder(dcgan)
                # self.noisy_nimute_vec = tf.add(self.G_encoder, z)
                self.noisy_nimute_vec = self.G_encoder
            else:
                self.noisy_nimute_vec = z
            self.G_decoder = self.generator_decoder(dcgan)

    def generator_encoder(self, dcgan):
        with tf.variable_scope("generator_encoder") as scope:
            if self.reuse:
                scope.reuse_variables()
            self.e_h0 = lrelu(conv2d(self.maps, dcgan.disc_input_layer_depth, name='g_ench0_conv'))
            self.e_h1 = lrelu(dcgan.d_bn1(conv2d(self.e_h0, dcgan.disc_input_layer_depth * 2, name='g_ench_1_conv')
                                          , train=self.train))
            self.e_h2 = lrelu(dcgan.d_bn2(conv2d(self.e_h1, dcgan.disc_input_layer_depth * 4, name='g_ench_2_conv')
                                          , train=self.train))
            self.e_h3 = lrelu(dcgan.d_bn3(conv2d(self.e_h2, dcgan.disc_input_layer_depth * 8, name='g_ench_3_conv')
                                          , train=self.train))
            self.e_h4 = lrelu(dcgan.d_bn4(conv2d(self.e_h3, dcgan.disc_input_layer_depth * 16, name='g_ench_4_conv')
                                          , train=self.train))
            self.e_h5 = lrelu(dcgan.d_bn5(conv2d(self.e_h4, dcgan.disc_input_layer_depth * 32, name='g_ench_5_conv')
                                          , train=self.train))
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
                tf.nn.relu(dcgan.g_bnz(self.z_1, train=self.train)), dcgan.gen_input_layer_depth * 32 * s_h64 * s_w64, 'g_h1_lin', with_w=True)

            self.d_h0 = tf.reshape(
                self.z_, [-1, s_h64, s_w64, dcgan.gen_input_layer_depth * 32])
            self.d_h0 = tf.nn.relu(dcgan.g_bn0(self.d_h0, train=self.train))

            self.d_h1, _, _ = deconv2d(
                self.d_h0, [dcgan.batch_size, s_h32, s_w32, dcgan.gen_input_layer_depth * 16], name='g_h1', with_w=True)
            self.d_h1 = tf.nn.relu(dcgan.g_bn1(self.d_h1, train=self.train))

            self.d_h2, _, _ = deconv2d(
                self.d_h1, [dcgan.batch_size, s_h16, s_w16, dcgan.gen_input_layer_depth * 8], name='g_h2', with_w=True)
            self.d_h2 = tf.nn.relu(dcgan.g_bn2(self.d_h2, train=self.train))

            self.d_h3, _, _ = deconv2d(
                self.d_h2, [dcgan.batch_size, s_h8, s_w8, dcgan.gen_input_layer_depth * 4], name='g_h3', with_w=True)
            self.d_h3 = tf.nn.relu(dcgan.g_bn3(self.d_h3, train=self.train))

            self.d_h4, _, _ = deconv2d(
                self.d_h3, [dcgan.batch_size, s_h4, s_w4, dcgan.gen_input_layer_depth * 2], name='g_h4', with_w=True)
            self.d_h4 = tf.nn.relu(dcgan.g_bn4(self.d_h4, train=self.train))

            self.d_h5, _, _ = deconv2d(
                self.d_h4, [dcgan.batch_size, s_h2, s_w2, dcgan.gen_input_layer_depth * 1], name='g_h5', with_w=True)
            self.d_h5 = tf.nn.relu(dcgan.g_bn5(self.d_h5, train=self.train))

            if dcgan.use_maps:
                self.d_h6, _, _ = deconv2d(
                    self.d_h5, [dcgan.batch_size, s_h, s_w, int(dcgan.gen_input_layer_depth * 0.5)], name='g_h6', with_w=True)
                self.d_h6 = tf.nn.relu(self.d_h6)
                self.d_h6 = tf.concat([self.d_h6, self.maps], axis=-1)

                self.d_h7 = conv2d(self.d_h6, dcgan.c_dim, k_h=3, k_w=3, s_h=1, s_w=1, name='g_h7')
                return tf.nn.tanh(self.d_h7)

            self.d_h6, _, _ = deconv2d(
                self.d_h5, [dcgan.batch_size, s_h, s_w, dcgan.c_dim], name='g_h6', with_w=True)

            return tf.nn.tanh(self.d_h6)


class Unet(Generator):
    def __init__(self, dcgan, z, maps=None, reuse=False, train=True, use_encoder=True):
        super(Unet, self).__init__(dcgan, z, maps=maps, reuse=reuse, train=train)
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
                tf.nn.relu(dcgan.g_bnz(self.z_1, train=self.train)), dcgan.gen_input_layer_depth * 32 * s_h64 * s_w64, 'g_h1_lin', with_w=True)

            self.d_h0 = tf.reshape(
                self.z_, [-1, s_h64, s_w64, dcgan.gen_input_layer_depth * 32])
            self.d_h0 = tf.nn.relu(dcgan.g_bn0(self.d_h0, train=self.train))

            self.d_h1, _, _ = deconv2d(
                tf.concat([self.e_h5, self.d_h0], -1), [dcgan.batch_size, s_h32, s_w32, dcgan.gen_input_layer_depth * 16], name='g_h1', with_w=True)
            self.d_h1 = tf.nn.relu(dcgan.g_bn1(self.d_h1, train=self.train))

            self.d_h2, _, _ = deconv2d(
                tf.concat([self.e_h4, self.d_h1], -1), [dcgan.batch_size, s_h16, s_w16, dcgan.gen_input_layer_depth * 8], name='g_h2', with_w=True)
            self.d_h2 = tf.nn.relu(dcgan.g_bn2(self.d_h2, train=self.train))

            self.d_h3, _, _ = deconv2d(
                tf.concat([self.e_h3, self.d_h2], -1), [dcgan.batch_size, s_h8, s_w8, dcgan.gen_input_layer_depth * 4], name='g_h3', with_w=True)
            self.d_h3 = tf.nn.relu(dcgan.g_bn3(self.d_h3, train=self.train))

            self.d_h4, _, _ = deconv2d(
                tf.concat([self.e_h2, self.d_h3], -1), [dcgan.batch_size, s_h4, s_w4, dcgan.gen_input_layer_depth * 2], name='g_h4', with_w=True)
            self.d_h4 = tf.nn.relu(dcgan.g_bn4(self.d_h4, train=self.train))

            self.d_h5, _, _ = deconv2d(
                tf.concat([self.e_h1, self.d_h4], -1), [dcgan.batch_size, s_h2, s_w2, dcgan.gen_input_layer_depth * 1], name='g_h5', with_w=True)
            self.d_h5 = tf.nn.relu(dcgan.g_bn5(self.d_h5, train=self.train))

            self.d_h6, _, _ = deconv2d(
                tf.concat([self.e_h0, self.d_h5], -1), [dcgan.batch_size, s_h, s_w, int(dcgan.gen_input_layer_depth * 0.5)], name='g_h6', with_w=True)
            self.d_h6 = tf.nn.relu(self.d_h6)
            self.d_h6 = tf.concat([self.d_h6, self.maps], axis=-1)

            self.d_h7 = conv2d(self.d_h6, dcgan.c_dim, k_h=3, k_w=3, s_h=1, s_w=1, name='g_h7')
            return tf.nn.tanh(self.d_h7)


class ResizeAndConv(Generator):
    def __init__(self, dcgan, z, maps=None, reuse=False, train=True, use_encoder=True):
        super(ResizeAndConv, self).__init__(dcgan, z, maps=maps, reuse=reuse, train=train)
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
                tf.nn.relu(dcgan.g_bnz(self.z_1, train=self.train)), dcgan.gen_input_layer_depth * 32 * s_h64 * s_w64, 'g_h1_lin', with_w=True)

            self.d_h0 = tf.reshape(
                self.z_, [-1, s_h64, s_w64, dcgan.gen_input_layer_depth * 32])
            self.d_h0 = tf.nn.relu(dcgan.g_bn0(self.d_h0, train=self.train))

            self.d_h1 = resize_and_conv(
                self.d_h0, [dcgan.batch_size, s_h32, s_w32, dcgan.gen_input_layer_depth * 16], name='g_h1')
            self.d_h1 = tf.nn.relu(dcgan.g_bn1(self.d_h1, train=self.train))

            self.d_h2 = resize_and_conv(
                self.d_h1, [dcgan.batch_size, s_h16, s_w16, dcgan.gen_input_layer_depth * 8], name='g_h2')
            self.d_h2 = tf.nn.relu(dcgan.g_bn2(self.d_h2, train=self.train))

            self.d_h3 = resize_and_conv(
                self.d_h2, [dcgan.batch_size, s_h8, s_w8, dcgan.gen_input_layer_depth * 4], name='g_h3')
            self.d_h3 = tf.nn.relu(dcgan.g_bn3(self.d_h3, train=self.train))

            self.d_h4 = resize_and_conv(
                self.d_h3, [dcgan.batch_size, s_h4, s_w4, dcgan.gen_input_layer_depth * 2], name='g_h4')
            self.d_h4 = tf.nn.relu(dcgan.g_bn4(self.d_h4, train=self.train))

            self.d_h5 = resize_and_conv(
                self.d_h4, [dcgan.batch_size, s_h2, s_w2, dcgan.gen_input_layer_depth * 1], name='g_h5')
            self.d_h5 = tf.nn.relu(dcgan.g_bn5(self.d_h5, train=self.train))

            if dcgan.use_maps:
                self.d_h6 = resize_and_conv(
                    self.d_h5, [dcgan.batch_size, s_h, s_w, int(dcgan.gen_input_layer_depth * 0.5)], name='g_h6')
                self.d_h6 = tf.nn.relu(self.d_h6)
                self.d_h6 = tf.concat([self.d_h6, self.maps], axis=-1)

                self.d_h7 = conv2d(self.d_h6, dcgan.c_dim, k_h=3, k_w=3, s_h=1, s_w=1, name='g_h7')
                return tf.nn.tanh(self.d_h7)

            self.d_h6 = resize_and_conv(
                self.d_h5, [dcgan.batch_size, s_h, s_w, dcgan.c_dim], name='g_h6')

            return tf.nn.tanh(self.d_h6)


generators = {"decoder": Generator,
              "encoder_decoder": Generator,
              "unet": Unet,
              "resize_and_conv": ResizeAndConv}


def generator_factory(dcgan, z, maps=None, reuse=False, train=True, generator_name="encoder_decoder"):
    if generator_name not in generators.keys():
        assert ('Invalid generator name! {} is not know. Valid generators: {}'.format(generator_name, generators.keys()))
    if generator_name == "decoder":
        use_encoder = False
    else:
        use_encoder = True
    net = generators[generator_name](dcgan, z, maps, reuse=reuse, train=train, use_encoder=use_encoder)
    return net.G_decoder
