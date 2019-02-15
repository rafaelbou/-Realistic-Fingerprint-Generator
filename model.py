from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from pre_process import *
from nets.generators import generator_factory
from nets.discriminators import discriminator_factory
from objectives import sigmoid_cross_entropy_with_logits, l2_loss_weighted


class DCGAN(object):
    def __init__(self, sess, input_height=650, input_width=650, crop=True,
                 batch_size=4, sample_num=64, output_height=650, output_width=650,
                 z_dim=100, maps_dim=3, use_maps_flag=True, use_mask_flag=False, gen_input_layer_depth=8,
                 disc_input_layer_depth=8, gen_fc_size=1024, disc_fc_size=1024, dataset_name='default',
                 dataset_images_name='default', dataset_labels_name='default', dataset_masks_name='default',
                 input_fname_pattern='*.jpg', labels_fname_pattern='*.txt', masks_fname_pattern='*.png',
                 checkpoint_dir=None, data_dir='./data', logs_dir='./logs', load_samples_mode='validation', lamda=100.,
                 generator_model_name="encoder_decoder"):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          gen_input_layer_depth: (optional) Dimension of gen filters in first conv layer. [64]
          disc_input_layer_depth: (optional) Dimension of discrim filters in first conv layer. [64]
          gen_fc_size: (optional) Dimension of gen units for for fully connected layer. [1024]
          disc_fc_size: (optional) Dimension of discrim units for fully connected layer. [1024]
        """
        self.sess = sess
        # Data
        self.sample_num = sample_num
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.crop = crop
        # Hyper-params
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.maps_dim = maps_dim
        self.use_maps = use_maps_flag
        self.use_mask = use_mask_flag
        self.gen_input_layer_depth = gen_input_layer_depth
        self.disc_input_layer_depth = disc_input_layer_depth
        self.gen_fc_size = gen_fc_size
        self.disc_fc_size = disc_fc_size
        self.lamda = lamda
        # batch normalization: deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')
        self.g_bnz = batch_norm(name='g_bnz')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')
        # IO
        self.dataset_name = dataset_name
        self.dataset_images_name = dataset_images_name
        self.dataset_labels_name = dataset_labels_name
        self.dataset_masks_name = dataset_masks_name
        self.input_fname_pattern = input_fname_pattern
        self.labels_fname_pattern = labels_fname_pattern
        self.masks_fname_pattern = masks_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.logs_dir = logs_dir
        self.load_samples_mode = load_samples_mode
        self.save_inputs = True
        # Read dataset files
        self.read_dataset_files()
        # Build model
        self.generator_model_name = generator_model_name
        self.build_model()

    def read_dataset_files(self):
        # Read dataset files
        data_path = os.path.join(self.data_dir, self.dataset_name, self.load_samples_mode, self.dataset_images_name,
                                 self.input_fname_pattern)
        self.data = glob(data_path)
        if len(self.data) == 0:
            raise Exception("[!] No data found in '" + data_path + "'")
        np.random.shuffle(self.data)
        imreadImg = imread(self.data[0])
        if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1
        if len(self.data) < self.batch_size:
            raise Exception("[!] Entire dataset size is less than the configured batch_size")

        self.grayscale = (self.c_dim == 1)

    def pre_process(self):
        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]
        return image_dims

    def build_model(self):
        # input
        image_dims = self.pre_process()
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        inputs = self.inputs
        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        if self.use_maps:
            self.maps = tf.placeholder(
                tf.float32, [self.batch_size, self.output_height, self.output_width, self.maps_dim], name='maps')

        # build model
        if self.use_maps:
            self.G = generator_factory(self, self.z, self.maps, generator_name=self.generator_model_name)
        else:
            self.G = generator_factory(self, self.z, generator_name='decoder')
        self.D, self.D_logits = discriminator_factory(self, inputs, reuse=False)
        if self.use_maps:
            self.sampler = generator_factory(self, self.z, self.maps, reuse=True, generator_name=self.generator_model_name)
        else:
            self.sampler = generator_factory(self, self.z, reuse=True, generator_name='decoder')
        self.D_, self.D_logits_ = discriminator_factory(self, self.G, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        if self.use_maps:
            weights = tf.expand_dims(tf.math.add(tf.math.add(self.maps[:, :, :, 0], self.maps[:, :, :, 1]) * 10,
                                                 self.maps[:, :, :, 2] * 100), axis=-1)
            self.l2_g_loss = l2_loss_weighted(self.G, inputs, tf.math.add(weights, tf.ones_like(weights)))
            self.g_loss = self.g_loss + self.lamda * self.l2_g_loss

        # add summary
        self.add_summary()

        # create var lists
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # model saver
        self.saver = tf.train.Saver()

    def add_summary(self):
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.g_l2_loss_sum = scalar_summary("l2_g_loss", self.l2_g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
        self.d_sum = histogram_summary("d", self.D)
        self.z_sum = histogram_summary("z", self.z)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)
        if self.use_maps:
            self.mnt_sum = image_summary("G", self.maps)
            self.inputs_sum = image_summary("G", self.inputs)
            self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                        self.G_sum, self.mnt_sum, self.inputs_sum, self.d_loss_fake_sum,
                                        self.g_loss_sum, self.g_l2_loss_sum])
        else:
            self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                        self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter(self.logs_dir, self.sess.graph)

    def train(self, config):
        d_optim, g_optim = self.create_optimizer(config)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # load samples
        if self.use_maps:
            sample_inputs, sample_z, sample_maps = self.sample_inputs_and_z()
        else:
            sample_inputs, sample_z = self.sample_inputs_and_z()
        counter = self.load(self.checkpoint_dir)

        # run epochs
        start_time = time.time()
        for epoch in xrange(config.epoch):
            data_mode = 'train'
            data_path = os.path.join(self.data_dir, self.dataset_name, data_mode, self.dataset_images_name,
                                     self.input_fname_pattern)
            self.data = glob(os.path.join(data_path))
            np.random.shuffle(self.data)
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, int(batch_idxs)):
                batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                if self.use_mask:
                    masks = [batch_file.replace(self.dataset_images_name, self.dataset_masks_name)
                                 .replace(self.input_fname_pattern[1:], self.masks_fname_pattern[1:])
                             for batch_file in batch_files]
                else:
                    masks = [None for _ in batch_files]
                batch = [
                    get_image(batch_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              crop=self.crop,
                              grayscale=self.grayscale,
                              mask=masks[batch_idx]) for batch_idx, batch_file in enumerate(batch_files)]
                if self.use_maps:
                    batch_maps = [
                        get_label(batch_file.replace(self.dataset_images_name, self.dataset_labels_name)
                                  .replace(self.input_fname_pattern[1:], self.labels_fname_pattern[1:]),
                                  input_height=self.input_height,
                                  input_width=self.input_width,
                                  resize_height=self.output_height,
                                  resize_width=self.output_width,
                                  crop=self.crop,
                                  mask=masks[batch_idx]) for batch_idx, batch_file in enumerate(batch_files)]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                if self.use_maps:
                    # w_test = tf.math.add(self.maps, tf.ones_like(self.maps))
                    # test = self.sess.run(w_test, feed_dict={self.maps: batch_maps})
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.inputs: batch_images, self.z: batch_z,
                                                              self.maps: batch_maps})
                else:
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                if self.use_maps:
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z, self.maps: batch_maps,
                                                              self.inputs: batch_images})
                else:
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z})
                if idx % config.summary_steps == 0:
                    self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                if self.use_maps:
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z, self.maps: batch_maps,
                                                              self.inputs: batch_images})
                else:
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z})
                self.eval_and_save(batch_idxs, batch_images, batch_z, config, counter, epoch, idx, sample_inputs,
                                   sample_z, start_time, summary_str, batch_maps, sample_maps)
                counter += 1

    def eval_and_save(self, batch_idxs, batch_images, batch_z, config, counter, epoch, idx, sample_inputs, sample_z,
                      start_time, summary_str, batch_maps=None, sample_maps=None):
        if idx % config.summary_steps == 0:
            self.writer.add_summary(summary_str, counter)
        if self.use_maps:
            errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.maps: batch_maps})
        else:
            errD_fake = self.d_loss_fake.eval({self.z: batch_z})
        errD_real = self.d_loss_real.eval({self.inputs: batch_images})
        if self.use_maps:
            errG = self.g_loss.eval({self.z: batch_z, self.maps: batch_maps, self.inputs: batch_images})
            errL2G = self.l2_g_loss.eval({self.z: batch_z, self.maps: batch_maps, self.inputs: batch_images})
        else:
            errG = self.g_loss.eval({self.z: batch_z})
            errL2G = -1.
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, g_l2_loss: %.8f" \
              % (epoch, config.epoch, idx, batch_idxs,
                 time.time() - start_time, errD_fake + errD_real, errG, errL2G))
        if np.mod(counter, config.eval_steps) == 0:
            try:
                if self.use_maps:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_z,
                            self.inputs: sample_inputs,
                            self.maps: sample_maps
                        },
                    )
                else:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_z,
                            self.inputs: sample_inputs,
                        },
                    )
                if self.save_inputs:
                    save_images(sample_inputs, image_manifold_size(samples.shape[0]),
                                './{}/inputs.png'.format(config.sample_dir, epoch, idx))
                    self.save_inputs = False
                if not self.use_maps:
                    sample_maps = None
                save_images(samples, image_manifold_size(samples.shape[0]),
                            './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx), maps=sample_maps)
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
            except:
                print("one pic error!...")
        if np.mod(counter, config.save_ckpt_steps) == 0:
            self.save(config.checkpoint_dir, counter)

    def sample_inputs_and_z(self):
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        sample_files = self.data[0:self.sample_num]
        if self.use_mask:
            masks = [sample_file.replace(self.dataset_images_name, self.dataset_masks_name)
                         .replace(self.input_fname_pattern[1:], self.masks_fname_pattern[1:])
                     for sample_file in sample_files]
        else:
            masks = [None for _ in sample_files]
        sample = [
            get_image(sample_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale,
                      mask=masks[sample_idx]) for sample_idx, sample_file in enumerate(sample_files)]
        if self.use_maps:
            sample_maps = [
                get_label(sample_file.replace(self.dataset_images_name, self.dataset_labels_name)
                          .replace(self.input_fname_pattern[1:], self.labels_fname_pattern[1:]),
                          input_height=self.input_height,
                          input_width=self.input_width,
                          resize_height=self.output_height,
                          resize_width=self.output_width,
                          crop=self.crop,
                          mask=masks[sample_idx]) for sample_idx, sample_file in enumerate(sample_files)]
        if self.grayscale:
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        if self.use_maps:
            return sample_inputs, sample_z, sample_maps
        return sample_inputs, sample_z

    def create_optimizer(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        return d_optim, g_optim

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            print(" [*] Load SUCCESS")
            return counter
        else:
            print(" [*] Failed to find a checkpoint")
            print(" [!] Load failed...")
            return 0
