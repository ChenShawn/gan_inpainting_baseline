import tensorflow as tf
from utils import batch_norm, coord_conv, attention
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib import slim


def build_lenet(input_op, is_training=True):
    params = {'kernel_size': 5, 'padding': 'same', 'use_bias': False, 'activation': None,
              'strides': 2, 'kernel_initializer': tf.contrib.layers.xavier_initializer_conv2d()}
    conv_1 = coord_conv(input_op, 32, name='conv_1', **params)
    conv_1 = tf.nn.leaky_relu(batch_norm(conv_1, is_training=is_training, name='bn_1'))
    conv_2 = tf.layers.conv2d(conv_1, 64, name='conv_2', **params)
    conv_2 = tf.nn.leaky_relu(batch_norm(conv_2, is_training=is_training, name='bn_2'))
    conv_3 = tf.layers.conv2d(conv_2, 128, name='conv_3', **params)
    conv_3 = tf.nn.leaky_relu(batch_norm(conv_3, is_training=is_training, name='bn_3'))

    params = {'kernel_size': 5, 'strides': 2, 'padding': 'same',  'activation': None,
              'use_bias': False, 'kernel_initializer': tf.contrib.layers.xavier_initializer_conv2d()}
    deconv_1 = tf.layers.conv2d_transpose(conv_3, 64, name='deconv_1', **params)
    deconv_1 = tf.nn.leaky_relu(batch_norm(deconv_1, is_training=is_training))
    deconv_2 = tf.layers.conv2d_transpose(tf.concat([deconv_1, conv_2], axis=-1), 32, name='deconv_2', **params)
    deconv_2 = tf.nn.leaky_relu(batch_norm(deconv_2, is_training=is_training, name='bn_4'))
    params['activation'] = None
    params['use_bias'] = True
    return tf.layers.conv2d_transpose(tf.concat([deconv_2, conv_1], axis=-1), 1, name='deconv_3', **params)


def build_lenet1(input_op, is_training=True):
    params = {'kernel_size': 5, 'padding': 'same', 'use_bias': False,
              'strides': 2, 'kernel_initializer': tf.contrib.layers.xavier_initializer_conv2d()}
    conv_1 = coord_conv(input_op, 32, name='conv_1', **params)
    conv_1 = tf.nn.relu(batch_norm(conv_1, is_training=is_training, name='bn_1'))
    conv_2 = tf.layers.conv2d(conv_1, 64, name='conv_2', **params)
    conv_2 = tf.nn.relu(batch_norm(conv_2, is_training=is_training, name='bn_2'))
    conv_3 = tf.layers.conv2d(conv_2, 128, name='conv_3', **params)
    conv_3 = tf.nn.relu(batch_norm(conv_3, is_training=is_training, name='bn_3'))

    params = {'kernel_size': 5, 'strides': 2, 'padding': 'same',  'activation': tf.nn.leaky_relu,
              'use_bias': False, 'kernel_initializer': tf.contrib.layers.xavier_initializer_conv2d()}
    deconv_1 = tf.layers.conv2d_transpose(conv_3, 64, name='deconv_1', **params)
    deconv_2 = tf.layers.conv2d_transpose(tf.concat([deconv_1, conv_2], axis=-1), 32, name='deconv_2', **params)
    deconv_2 = tf.nn.relu(batch_norm(deconv_2, is_training=is_training, name='bn_4'))
    params['activation'] = None
    return tf.layers.conv2d_transpose(tf.concat([deconv_2, conv_1], axis=-1), 1, name='deconv_3', **params)


def build_unet(input_op, is_training=True, num_channels=3):
    # Definition of encoder network
    conv_1 = coord_conv(input_op, 64, kernel_size=3, padding='same', name='conv_1', use_bias=False,
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_1 = tf.nn.relu(batch_norm(conv_1, is_training=is_training, name='bn_1'))
    conv_2 = tf.layers.conv2d(conv_1, 64, kernel_size=3, padding='same', name='conv_2', use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_2 = tf.nn.relu(batch_norm(conv_2, is_training=is_training, name='bn_2'))
    pool_1 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2)

    conv_3 = coord_conv(pool_1, 128, kernel_size=3, padding='same', name='conv_3', use_bias=False,
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_3 = tf.nn.relu(batch_norm(conv_3, is_training=is_training, name='bn_3'))
    conv_4 = tf.layers.conv2d(conv_3, 128, kernel_size=3, padding='same', name='conv_4', use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_4 = tf.nn.relu(batch_norm(conv_4, is_training=is_training, name='bn_4'))
    conv_5 = tf.layers.conv2d(conv_4, 128, kernel_size=3, padding='same', name='conv_5', use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_5 = tf.nn.relu(batch_norm(conv_5, is_training=is_training, name='bn_5'))
    pool_2 = tf.layers.max_pooling2d(conv_5, pool_size=2, strides=2)

    conv_6 = coord_conv(pool_2, 256, kernel_size=3, padding='same', name='conv_6', use_bias=False,
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

    return tf.layers.conv2d(conv_12, num_channels, kernel_size=3, padding='same', name='logits', use_bias=True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())


def build_lenet_generator(input_op, is_training=True, reuse=False, scope='LeGenerator', last_layer_activation=tf.sigmoid):
    """build_lenet_generator
    :param last_layer_activation: either tf.tanh or tf.sigmoid is recommended
    """
    with tf.variable_scope(scope, reuse=reuse):
        logits = build_lenet(input_op, is_training)
        return last_layer_activation(logits)


def build_lenet_policy(input_op, is_training=True, reuse=False, scope='LePolicy'):
    """build_lenet_policy"""
    with tf.variable_scope(scope, reuse=reuse):
        logits = build_lenet(input_op, is_training)
        probs = tf.nn.sigmoid(logits)
    return logits, probs


def build_unet_generator(input_op, is_training=True, num_channels=3, reuse=False, scope='UGenerator',
                         last_layer_activation=tf.sigmoid, **kwargs):
    """build_unet_generator
    :param is_training: should be set to False when evaluating
    :param last_layer_activation: either tf.tanh or tf.sigmoid is recommended
    """
    with tf.variable_scope(scope, reuse=reuse):
        logits = build_unet(input_op, is_training=is_training, num_channels=num_channels)
        # build attention

        return last_layer_activation(logits, **kwargs)


def build_unet_policy(input_op, is_training=True, scope='UPolicy', reuse=False):
    """build_policy
    :param is_training: should be set to False when testing or sampling
    :return: logits, probs

    This function use coordinate convolution in each head of the block
    The returned variable logits is used to calculate the policy gradient
    probs is used as a discrete distribution from which the behavior policy is sampled from
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Definition of encoder network
        conv_1 = coord_conv(input_op, 64, kernel_size=3, padding='same', name='conv_1', use_bias=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_1 = tf.nn.relu(batch_norm(conv_1, is_training=is_training, name='bn_1'))
        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=2, strides=2)

        conv_2 = coord_conv(pool_1, 128, kernel_size=3, padding='same', name='conv_2', use_bias=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_2 = tf.nn.relu(batch_norm(conv_2, is_training=is_training, name='bn_2'))
        pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2)

        conv_3 = coord_conv(pool_2, 256, kernel_size=3, padding='same', name='conv_3', use_bias=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_3 = tf.nn.relu(batch_norm(conv_3, is_training=is_training, name='bn_3'))

        # Definition of decoder network
        deconv_1 = tf.layers.conv2d_transpose(conv_3, filters=128, kernel_size=3, strides=2, padding='same',
                                              name='deconv_1', use_bias=True, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        concat_1 = tf.concat([deconv_1, conv_2], axis=-1, name='concat_1')
        conv_4 = tf.layers.conv2d(concat_1, 128, kernel_size=3, padding='same', name='conv_4', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_4 = tf.nn.relu(batch_norm(conv_4, is_training=is_training, name='bn_4'))

        deconv_2 = tf.layers.conv2d_transpose(conv_4, filters=64, kernel_size=3, strides=2, padding='same',
                                              name='deconv_2', use_bias=True, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        concat_2 = tf.concat([deconv_2, conv_1], axis=-1, name='concat_2')
        conv_5 = tf.layers.conv2d(concat_2, 64, kernel_size=3, padding='same', name='conv_5', use_bias=False,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_5 = tf.nn.relu(batch_norm(conv_5, is_training=is_training, name='bn_5'))

        # attention
        """

        x= tf.pad(conv_5, [[0, 0], [0, 0], [0, 0], [0, 0]])
        f = tf.layers.conv2d(inputs=x, filters=8,
                             kernel_size=1, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                             kernel_regularizer=None,
                             strides=1, use_bias=False)
        g = tf.layers.conv2d(inputs=x, filters=8,
                             kernel_size=1, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                             kernel_regularizer=None,
                             strides=1, use_bias=False)
        h = tf.layers.conv2d(inputs=x, filters=64,
                             kernel_size=1, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                             kernel_regularizer=None,
                             strides=1, use_bias=False)
        g = tf.reshape(g, shape=[g.shape[0], -1, g.shape[-1]])
        f = tf.reshape(f, shape=[f.shape[0], -1, f.shape[-1]])
        s = tf.matmul(g, f, transpose_b=True) # # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map

        h = tf.reshape(h, shape=[h.shape[0], -1, h.shape[-1]])
        o = tf.matmul(beta, h) # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
        x = gamma * o + x
                """
        logits = tf.layers.conv2d(conv_5, 1, kernel_size=3, padding='same', name='conv_6', use_bias=False,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        probs = tf.nn.softmax(batch_norm(logits, is_training=is_training, name='bn_6'))
        # logits = build_unet(input_op, is_training=is_training, num_channels=1)
        # I will predict mas
        #probs = attention(logits, ch=1, sn=False, reuse=reuse)
        #probs = tf.sigmoid(logits)
    return logits, probs


def build_dcgan_discriminator(input_op, is_training=True, scope='UDiscriminator', reuse=False,
                              block_num=4, min_filters=64, kernel_size=3, activation=tf.nn.leaky_relu):
    """build_dcgan_discriminator
    The network structure follows that of DCGAN except that
    1) one more conv before each stride=2 down-sampling conv layer;
    2) change kernel size from 5 to 3 since double 3x3 kernels make similar effect with 5x5
    """
    with tf.variable_scope(scope, reuse=reuse):
        net = input_op
        end_points = dict()
        for idx in range(block_num):
            net = tf.layers.conv2d(net, min_filters * (idx + 1), kernel_size=kernel_size, padding='same',
                                   name='conv_' + str(2 * idx), use_bias=False, activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            net = activation(batch_norm(net, is_training=is_training, name='bn_' + str(2 * idx)))
            net = tf.layers.conv2d(net, min_filters * (idx + 1), kernel_size=kernel_size, strides=2, padding='same',
                                   name='conv_' + str(2 * idx + 1), activation=None, use_bias=False,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            net = activation(batch_norm(net, is_training=is_training, name='bn_' + str(2 * idx + 1)))
            # end_points should be returned to calculate the
            end_points['pool_' + str(idx)] = net

        batch_size = net.get_shape().as_list()[0]
        net = tf.reshape(net, [batch_size, -1])
        logits = tf.layers.dense(net, 1, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return logits, end_points


def build_vgg_pretraned_model(input_op, is_training, reuse=None, num_channels=3):
    upsample_params = {'kernel_size': 3, 'strides': 2, 'padding': 'same', 'use_bias': False,
                       'kernel_initializer': tf.contrib.layers.xavier_initializer_conv2d()}
    conv_params = {'kernel_size': 3, 'padding': 'same', 'use_bias': False, 'activation': None,
                   'kernel_initializer': tf.contrib.layers.xavier_initializer_conv2d()}
    def build_vgg_upsample_block(input_op, concat, filters, name):
        with tf.variable_scope(name):
            upsampled = tf.layers.conv2d_transpose(input_op, filters, **upsample_params)
            upsampled = tf.nn.relu(batch_norm(upsampled, is_training=is_training, name='upbn_1'))
            net = tf.concat([upsampled, concat], axis=-1)
            net = tf.layers.conv2d(net, filters, **conv_params)
            net = tf.nn.relu(batch_norm(net, is_training=is_training, name='upbn_2'))
            conv_params['use_bias'] = True
            return tf.layers.conv2d(net, filters, **conv_params)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected], reuse=reuse):
            logits, end_points = vgg.vgg_16(input_op, is_training=True, spatial_squeeze=False)

        # Build decoder network (using UNet structure)
        # 1/8 of the original size
        with tf.variable_scope('Decoder', reuse=reuse):
            net = build_vgg_upsample_block(end_points['vgg_16/conv5/conv5_3'],
                                           end_points['vgg_16/conv4/conv4_3'],
                                           filters=512, name='upsample_1')
            net = tf.nn.relu(batch_norm(net, is_training=is_training, name='bn_1'))
            # 1/4 of the original size
            net = build_vgg_upsample_block(net, end_points['vgg_16/conv3/conv3_3'], 256, name='upsample_2')
            net = tf.nn.relu(batch_norm(net, is_training=is_training, name='bn_2'))
            # 1/2 of the original size
            net = build_vgg_upsample_block(net, end_points['vgg_16/conv2/conv2_2'], 128, name='upsample_3')
            net = tf.nn.relu(batch_norm(net, is_training=is_training, name='bn_3'))
            net = build_vgg_upsample_block(net, end_points['vgg_16/conv1/conv1_2'], 64, name='upsample_4')
            net = tf.layers.conv2d(net, num_channels, kernel_size=3, padding='same',
                                   use_bias=True, activation=tf.nn.relu)
    return net, end_points


def build_generator_with_pconv(input_op, size, reuse, scope='PCGenerator'):
    pass


def build_resnet_generator(input_op, size, reuse, scope='ResNetGenerator'):
    pass


def build_resnet_discriminator(input_op, scope='ResNetDiscriminator', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        pass