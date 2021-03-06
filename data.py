import numpy as np
import tensorflow as tf
import os
import json

from utils import random_mask, gaussian_blur_masks, dropout_masks


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
    data_path = 'D:\\毕业论文\\data\\CelebA\\img_align_celeba\\'

    def __init__(self, size=(218, 178), batch_size=32, num_epochs=50, mask_type='blur'):
        """CelebAReader constructor
        :param mask_type: either `block`, `dropout` or `blur`
        """
        file_xs = list()
        parse_images = ImageParser(size=size, suffix='jpg')
        self.batch_size = batch_size

        recursive_path(self.data_path, file_xs)
        data = tf.data.Dataset.from_tensor_slices((tf.constant(file_xs)))
        data = data.map(parse_images, num_parallel_calls=8)
        self.data = data.shuffle(buffer_size=1024).batch(batch_size).repeat(num_epochs)
        self.batch_xs = tf.reshape(self.data.make_one_shot_iterator().get_next(),
                                   shape=[batch_size, size[0], size[1], 3])
        if mask_type == 'block':
            blocked_pixel_value = tf.random_uniform([], minval=0.0, maxval=1.0)
            self.lossy_xs, mask = random_mask(self.batch_xs, blocked_pixel_value=blocked_pixel_value)
        elif mask_type == 'blur':
            self.lossy_xs, mask = gaussian_blur_masks(self.batch_xs)
        elif mask_type == 'dropout':
            self.lossy_xs, mask = dropout_masks(self.batch_xs)
        else:
            raise NotImplementedError
        self.mask = tf.reduce_max(mask, axis=-1, keepdims=True)


class MnistReader(object):
    data_path = 'D:\\毕业论文\\tensorflow-generative-model-collections-master\\data\\mnist.npz'

    def __init__(self, size=(28, 28), batch_size=256, num_epochs=500, type='train'):
        """Constructor of class MnistReader
        :param type: either `train` or `test`
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        with np.load(self.data_path) as f:
            x_train = np.expand_dims(f['x_' + type], axis=3).astype(np.float32)
            y_train = f['y_' + type].astype(np.int32)

        self.data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.data = self.data.shuffle(buffer_size=1024).batch(batch_size).repeat(num_epochs)
        self.batch_xs, self.batch_ys = self.data.make_one_shot_iterator().get_next()
        self.batch_xs = tf.reshape(self.batch_xs, [batch_size, size[0], size[1], 1])
        self.batch_xs = tf.image.resize_images(self.batch_xs, (32, 32))

        # blocked_pixel_value = tf.random_uniform([], minval=x_train.min(), maxval=x_train.max())
        self.lossy_xs, mask = random_mask(self.batch_xs, blocked_pixel_value=0.0)
        self.mask = tf.reduce_max(mask, axis=-1, keepdims=True)


class PlaceReader(object):
    data_path = 'D:\\毕业论文\\data\\places\\places.json'

    def __init__(self, size=(224, 224), batch_size=4, num_epochs=50, type='train', mask_type='blur'):
        """PlaceReader constructor
        :param type: either `train` or `test`
        :param mask_type: either `block`, `dropout` or `blur`
        """
        parse_images = ImageParser(size=size, suffix='jpg')
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        with open(self.data_path) as f:
            X = json.load(f)[type]

        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data = data.map(parse_images)
        self.data = data.shuffle(buffer_size=1024).batch(batch_size).repeat(num_epochs)
        self.batch_xs = tf.reshape(self.data.make_one_shot_iterator().get_next(),
                                   shape=[batch_size, size[0], size[1], 3])
        if mask_type == 'block':
            blocked_pixel_value = tf.random_uniform([], minval=0.0, maxval=1.0)
            self.lossy_xs, self.mask = random_mask(self.batch_xs, ratio=3,
                                                   blocked_pixel_value=blocked_pixel_value)
        elif mask_type == 'blur':
            self.lossy_xs, self.mask = gaussian_blur_masks(self.batch_xs, ratio=ratio)
        elif mask_type == 'dropout':
            self.lossy_xs, self.mask = dropout_masks(self.batch_xs, ratio=ratio)
        else:
            raise NotImplementedError

        # TODO: remove this!
        self.comp_images = self.lossy_xs * (1.0 - self.mask) + self.batch_xs * self.mask



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    reader = PlaceReader()

    with tf.Session() as sess:
        xs, ys, mask, comp = sess.run([reader.batch_xs, reader.lossy_xs, reader.mask, reader.comp_images])
        print(xs.shape, ys.shape, mask.shape, xs.min(), xs.max())

    for it in range(reader.batch_size):
        plt.figure()
        plt.subplot(221)
        plt.imshow(xs[it, :, :, :])
        plt.title('Clean image')
        plt.axis('off')
        plt.subplot(222)
        plt.title('Masked image')
        plt.imshow(ys[it, :, :, :])
        plt.axis('off')
        plt.subplot(223)
        plt.title('Mask')
        plt.imshow(mask[it, :, :, :])
        plt.axis('off')
        plt.subplot(224)
        plt.title('Comp')
        plt.imshow(comp[it, :, :, :])
        plt.axis('off')
        plt.show()