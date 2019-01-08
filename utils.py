import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np


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
    horizontal = np.array([list(range(width)) for _ in range(height)], dtype=np.float32)
    vertical = np.array([list(range(height)) for _ in range(width)], dtype=np.float32)
    coords = np.concatenate([horizontal[:, :, None], vertical[:, :, None]], axis=-1)
    coords = (coords - coords.mean()) / coords.max()
    coords_array = np.array([coords[None, :, :, :] for _ in batch_size], dtype=np.float32)

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