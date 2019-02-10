import os
import numpy as np

from model import DCGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

flags = tf.app.flags
# IO
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("summary_steps", 100, "write to summery file each summary_steps steps")
flags.DEFINE_integer("eval_steps", 100, "run evaluation each eval_steps steps")
flags.DEFINE_integer("save_ckpt_steps", 100, "save checkpoint file each save_ckpt_steps steps")
# Data
flags.DEFINE_string("dataset", "NIST14", "The name of dataset [NIST14, FVC]")
flags.DEFINE_string("dataset_images", "Fimages", "The name of dataset [Fimages, Rimages]")
flags.DEFINE_string("dataset_labels", "Fiso", "The name of dataset [Fiso, Riso]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("labels_fname_pattern", "*.txt", "Glob pattern of filename of input images [*]")
flags.DEFINE_integer("input_height", 650, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 650, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
# Mode
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
# Hyper-params
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("lamda", 10., "Weight for l2 loss")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 4, "The size of batch images [64]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height
    if FLAGS.train:
        load_samples_mode = 'validation'
    else:
        load_samples_mode = 'test'

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    # run_config.gpu_options.allow_growth = True
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            z_dim=FLAGS.generate_test_images,
            dataset_name=FLAGS.dataset,
            dataset_images_name=FLAGS.dataset_images,
            dataset_labels_name=FLAGS.dataset_labels,
            input_fname_pattern=FLAGS.input_fname_pattern,
            labels_fname_pattern=FLAGS.labels_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            data_dir=FLAGS.data_dir,
            load_samples_mode=load_samples_mode,
            lamda=FLAGS.lamda)

        show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")

        # visualization code run both in train/test mode.
        visualize(sess, dcgan, FLAGS)


if __name__ == '__main__':
    tf.app.run()
