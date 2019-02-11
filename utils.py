import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import math
import os
import re
import extra


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


def build_total_variation_loss(image):
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
    return tf.reduce_mean(tf.abs(dxdy))


def build_style_loss(input_xs, input_ys):
    batch_size, height, width, channel = input_xs.get_shape().as_list()
    matrix_xs = tf.reshape(input_xs, [batch_size, -1, channel])
    matrix_ys = tf.reshape(input_ys, [batch_size, -1, channel])
    K_p = 1.0 / ((float(height) * float(width) * float(channel)) ** 2)
    style_diff = tf.matmul(matrix_xs, matrix_xs, transpose_a=True) - \
                 tf.matmul(matrix_ys, matrix_ys, transpose_a=True)
    return tf.reduce_mean(tf.abs(K_p * style_diff))


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
    return results, masking


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


def build_psnr(generated, ground_truth):
    """build_psnr
    TODO: This function is still nemerically unstable
    """
    mse = tf.square(generated - ground_truth)

    return tf.log(generated + 1e-25) / math.log(10.0)


def attention(x, ch, sn, reuse):
    """
    :param x: input_op
    :param ch: input channel
    :param sn: isNorm
    :param reuse
    :return:
    """
    with tf.variable_scope('attention', reuse=reuse):
        f = extra.ops.conv(x, ch, kernel=1, stride=1, sn=sn, scope='f_conv') # [bs, h, w, c']
        g = extra.ops.conv(x, ch, kernel=1, stride=1, sn=sn, scope='g_conv') # [bs, h, w, c']
        h = extra.ops.conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv') # [bs, h, w, c]
        #  N = h * w
        s = tf.matmul(extra.ops.hw_flatten(g), extra.ops.hw_flatten(f), transpose_b=True) # # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map
        o = tf.matmul(beta, extra.ops.hw_flatten(h)) # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
        x = gamma * o + x
    return x

