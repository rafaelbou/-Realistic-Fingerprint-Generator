import scipy.misc
import numpy as np
from utils import imread
from preprocess.create_minute_map import create_label_map


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False, mask=None):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop, mask=mask)


def get_label(label_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, mask=None):
    image = create_label_map(label_path)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop, label=True, mask=mask)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def mask_crop(x, mask, resize_h=64, resize_w=64):
    import cv2
    cc = cv2.findContours(mask.astype(np.uint8)[:, :, np.newaxis], mode=cv2.RETR_EXTERNAL,
                          method=cv2.CHAIN_APPROX_SIMPLE)[1][-1]
    left = max(0, cc[:, :, 0].min() - 10)
    top = max(0, cc[:, :, 1].min() - 10)
    right = min(cc[:, :, 0].max() + 10, x.shape[1])
    bottom = min(cc[:, :, 1].max() + 10, x.shape[0])
    output_img = x[top:bottom, left:right]
    output_img = scipy.misc.imresize(output_img, [resize_h, resize_w])
    return output_img


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True, label=False, mask=None):
    if crop:
        if mask is None:
            cropped_image = center_crop(
                image, input_height, input_width,
                resize_height, resize_width)
        else:
            mask = imread(mask)
            cropped_image = mask_crop(image, mask, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    if label:
        return np.array(cropped_image) / 255.
    return np.array(cropped_image) / 127.5 - 1.
