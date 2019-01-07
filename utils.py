import tensorflow as tf
from tensorflow.contrib import slim


def show_all_variables():
    all_variables = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(all_variables, print_info=True)


def batch_norm(input_op, is_training, epsilon=1e-5, momentum=0.99, name='batch_norm'):
    return tf.contrib.layers.batch_norm(input_op,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=is_training,
                                        scope=name)


def coord_conv(input_op, input_size, *args, **kwargs):
    pass