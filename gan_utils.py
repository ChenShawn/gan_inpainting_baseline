import tensorflow as tf
from utils import batch_norm


def build_unet_generator(input_op, is_training, num_channels=3, reuse=False, scope='UGenerator'):
    with tf.variable_scope(scope, reuse=reuse):
        # Definition of encoder network
        conv_1 = tf.layers.conv2d(input_op, 64, kernel_size=3, padding='same', name='conv_1', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_1 = tf.nn.relu(batch_norm(conv_1, is_training=is_training, name='bn_1'))
        conv_2 = tf.layers.conv2d(conv_1, 64, kernel_size=3, padding='same', name='conv_2', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_2 = tf.nn.relu(batch_norm(conv_2, is_training=is_training, name='bn_2'))

        pool_1 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2)
        conv_3 = tf.layers.conv2d(pool_1, 128, kernel_size=3, padding='same', name='conv_3', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_3 = tf.nn.relu(batch_norm(conv_3, is_training=is_training, name='bn_3'))
        conv_4 = tf.layers.conv2d(conv_3, 128, kernel_size=3, padding='same', name='conv_4', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_4 = tf.nn.relu(batch_norm(conv_4, is_training=is_training, name='bn_4'))
        conv_5 = tf.layers.conv2d(conv_4, 128, kernel_size=3, padding='same', name='conv_5', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_5 = tf.nn.relu(batch_norm(conv_5, is_training=is_training, name='bn_5'))

        pool_2 = tf.layers.max_pooling2d(conv_5, pool_size=2, strides=2)
        conv_6 = tf.layers.conv2d(pool_2, 256, kernel_size=3, padding='same', name='conv_6', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_6 = tf.nn.relu(batch_norm(conv_6, is_training=is_training, name='bn_6'))
        conv_7 = tf.layers.conv2d(conv_6, 256, kernel_size=3, padding='same', name='conv_7', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_7 = tf.nn.relu(batch_norm(conv_7, is_training=is_training, name='bn_7'))
        conv_8 = tf.layers.conv2d(conv_7, 256, kernel_size=3, padding='same', name='conv_8', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_8 = tf.nn.relu(batch_norm(conv_8, is_training=is_training, name='bn_8'))

        # Definition of decoder network
        deconv_1 = tf.layers.conv2d_transpose(conv_8, filters=128, kernel_size=3, strides=2, padding='same',
                                              name='deconv_1', use_bias=True, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        concat_1 = tf.concat([deconv_1, conv_5], axis=-1, name='concat_1')
        conv_9 = tf.layers.conv2d(concat_1, 128, kernel_size=3, padding='same', name='conv_9', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_9 = tf.nn.relu(batch_norm(conv_9, is_training=is_training, name='bn_9'))
        conv_10 = tf.layers.conv2d(conv_9, 128, kernel_size=3, padding='same', name='conv_10', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_10 = tf.nn.relu(batch_norm(conv_10, is_training=is_training, name='bn_10'))

        deconv_2 = tf.layers.conv2d_transpose(conv_10, filters=64, kernel_size=3, strides=2, padding='same',
                                              name='deconv_2', use_bias=True, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        concat_2 = tf.concat([deconv_2, conv_2], axis=-1, name='concat_2')
        conv_11 = tf.layers.conv2d(concat_2, 64, kernel_size=3, padding='same', name='conv_11', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_11 = tf.nn.relu(batch_norm(conv_11, is_training=is_training, name='bn_11'))
        conv_12 = tf.layers.conv2d(conv_11, 64, kernel_size=3, padding='same', name='conv_12', use_bias=False,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_12 = tf.nn.relu(batch_norm(conv_12, is_training=is_training, name='bn_12'))

        logits = tf.layers.conv2d(conv_12, num_channels, kernel_size=3, padding='same', name='logits', use_bias=True,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    return logits, preds


def build_generator_with_pconv(input_op, size, reuse, scope='PCGenerator'):
    pass


def build_resnet_generator(input_op, size, reuse, scope='ResNetGenerator'):
    pass


def build_discriminator():
    pass


def build_policy(input_op, is_training, num_channels=3, scope='Policy', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        pass