import numpy as np
import tensorflow as tf
import os

from utils import random_mask


class ImageParser(object):
    def __init__(self, size, dtype=tf.float32, suffix='jpg', method=tf.image.ResizeMethod.BILINEAR):
        """ImageParser constructor
        :param size: list or tuple of length 2, the new size of the images
        :param dtype: tf.float32 or tf.float32 is recommended
        :param suffix: either `png` or `jpg`
        """
        self.size = size
        self.dtype = dtype
        self.suffix = suffix
        self.method = method

    def __call__(self, filename):
        x_img_str = tf.read_file(filename)
        if self.suffix == 'jpg':
            x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(x_img_str), self.dtype)
        else:
            x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_png(x_img_str), self.dtype)
        return tf.image.resize_images(x_img_decoded, size=self.size, method=self.method)



def parse_paired_images(filename, labelname, size, suffix='png', dtype_xs=tf.float32, dtype_ys=tf.float32,
                        method_xs=tf.image.ResizeMethod.BILINEAR,
                        method_ys=tf.image.ResizeMethod.BILINEAR):
    """parse_paired_images
        Either set both dtype_ys and dtype_xs=tf.float32 for inpainting tasks
        or set dtype_ys=tf.int32 and method_ys=tf.images.ResizeMethod.NEAREST_NEIGHBOR if y is label
    """
    x_img_str = tf.read_file(filename)
    if suffix == 'png':
        x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_png(x_img_str), dtype_xs)
    else:
        x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(x_img_str), dtype_xs)
    x_img_resized = tf.image.resize_images(x_img_decoded, size=size, method=method_xs)

    y_img_str = tf.read_file(labelname)
    if suffix == 'png':
        y_img_decoded = tf.image.convert_image_dtype(tf.image.decode_png(y_img_str), dtype_ys)
    else:
        y_img_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(y_img_str), dtype_ys)
    y_img_resized = tf.image.resize_images(y_img_decoded, size=size, method=method_ys)
    return x_img_resized, y_img_resized


def recursive_path(path, result, word=None):
    """recursive_path
    :param path: type str
    :param result: type list, returned as the results for recursion
    :param word: type str, filtering the results
    """
    fs = os.listdir(path)
    for f in fs:
        filename = os.path.join(path, f)
        if os.path.isdir(filename):
            recursive_path(filename, result)
        else:
            if word is not None:
                if word in filename:
                    result.append(filename)
            else:
                result.append(filename)



class CelebAReader(object):
    data_path = ''

    def __init__(self, size=(218, 178), batch_size=32, num_epochs=50):
        file_xs = list()
        parse_images = ImageParser(size=size, suffix='jpg')
        self.batch_size = batch_size

        recursive_path(self.data_path, file_xs)
        data = tf.data.Dataset.from_tensor_slices((tf.constant(file_xs)))
        data = data.map(parse_images)
        self.data = data.shuffle(buffer_size=1024).batch(batch_size).repeat(num_epochs)
        self.batch_xs = tf.reshape(self.data.make_one_shot_iterator().get_next(),
                                   shape=[batch_size, size[0], size[1], -1])

        self.lossy_xs = random_mask(self.batch_xs, blocked_pixel_value=0.0001)