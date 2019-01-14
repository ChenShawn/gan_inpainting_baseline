import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os, re


def show_all_variables(scope=None):
    all_variables = tf.trainable_variables(scope=scope)
    slim.model_analyzer.analyze_vars(all_variables, print_info=True)


def batch_norm(input_op, is_training, epsilon=1e-5, momentum=0.99, name='batch_norm'):
    return tf.contrib.layers.batch_norm(input_op,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=is_training,
                                        scope=name)


def coord_conv(input_op, *args, **kwargs):
    """coord_conv
    :param *args, **kwargs: params required to pass to function tf.layers.conv2d
    """
    batch_size, width, height, _ = input_op.get_shape().as_list()
    horizontal = np.array([list(range(width)) for _ in range(height)], dtype=np.float32).transpose()
    vertical = np.array([list(range(height)) for _ in range(width)], dtype=np.float32)
    coords = np.concatenate([horizontal[:, :, None], vertical[:, :, None]], axis=-1)
    coords = (coords - coords.mean()) / coords.max()
    coords_array = np.concatenate([coords[None, :, :, :] for _ in range(batch_size)], axis=0)

    coords_tensor = tf.convert_to_tensor(coords_array, dtype=tf.float32)
    input_tensor = tf.concat([input_op, coords_tensor], axis=-1)
    return tf.layers.conv2d(input_tensor, *args, **kwargs)


def total_variation_loss(image):
    """total_variation_loss using tensorflow internal ops"""
    # Construct first derivative conv kernels along x and y axis
    batch_size, height, width, channel = image.get_shape().as_list()
    horizontal, vertical = [np.zeros((3, 3, channel, 1), dtype=np.float32)] * 2
    horizontal[1, 2, :, :] = 1.0
    horizontal[1, 1, :, :] = -1.0
    vertical[2, 1, :, :] = 1.0
    vertical[1, 1, :, :] = -1.0
    kernel_array = np.concatenate([horizontal, vertical], axis=-1)

    # Convert them into tf tensors and convolve
    kernel_tensor = tf.constant(kernel_array, dtype=tf.float32)
    dxdy = tf.nn.conv2d(image, kernel_tensor, (1, 1, 1, 1), padding='SAME')
    return tf.reduce_sum(tf.abs(dxdy))


def random_mask(images, ratio=2, blocked_pixel_value=0.0):
    batch_size, height, width, channels = images.get_shape().as_list()
    mask_shape = [1, height // ratio, width // ratio, channels]
    with tf.name_scope('RandomMask'):
        off_h = tf.random_uniform([batch_size], 0, (height // ratio) * (ratio - 1), dtype=tf.int32)
        off_w = tf.random_uniform([batch_size], 0, (width // ratio) * (ratio - 1), dtype=tf.int32)
        masks = [tf.ones(mask_shape, dtype=tf.float32) for _ in range(batch_size)]
        paddings = [tf.image.pad_to_bounding_box(masks[it], off_h[it], off_w[it], height, width)
                    for it in range(batch_size)]

        mask_tensor = tf.concat(paddings, axis=0)
        masking = tf.cast(mask_tensor, dtype=tf.float32)
        results = images * (1.0 - masking) + masking * blocked_pixel_value
    return results, mask_tensor


def save(sess, model_path, model_name, global_step, remove_previous_files=True):
    saver = tf.train.Saver()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    elif len(os.listdir(model_path)) != 0 and remove_previous_files:
        fs = os.listdir(model_path)
        for f in fs:
            os.remove(os.path.join(model_path, f))

    saved_path = saver.save(sess, os.path.join(model_path, model_name), global_step=global_step)
    print('MODEL SAVED IN: ' + saved_path)
    return saved_path


def load(sess, model_path):
    print(" [*] Reading checkpoints...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_path, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator