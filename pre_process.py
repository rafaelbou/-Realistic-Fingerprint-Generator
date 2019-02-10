import scipy.misc
import numpy as np
from utils import imread
from preprocess.create_minute_map import create_label_map


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def get_label(label_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True):
    image = create_label_map(label_path)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop, label=True)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True, label=False):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    if label:
        return np.array(cropped_image) / 255.
    return np.array(cropped_image) / 127.5 - 1.
