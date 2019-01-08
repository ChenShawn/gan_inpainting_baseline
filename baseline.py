import tensorflow as tf
import numpy as np
import argparse

from gan_utils import *
from utils import show_all_variables, total_variation_loss

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='celeba', help='support only CelebA currently')
parser.add_argument('-f', '--function', type=str, default='train', help='`pretrain`, `train` or `eval`')
parser.add_argument('-l', '--logdir', type=str, default='./logs', help='dir for tensorboard logs')
parser.add_argument('-s', '--save_dir', type=str, default='./ckpt', help='dir for checkpoints')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-i', '--iterations', type=int, default=20000, help='number of training iterations')
parser.add_argument('-p', '--pretrain', type=bool, default=True, help='whether to load model from ckpt')
parser.add_argument('-a', '--allow_growth', type=bool, default=True, help='whether to grab all gpu resources')
parser.add_argument('--gp_lambda', type=float, default=0.25, help='coefficient for GP loss term')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning_rate')
parser.add_argument('--alpha', type=float, default=0.05, help='initial value of perceptual loss coefficient')
args = parser.parse_args()


class BaselineModel(object):
    name = 'BaselineModel'

    def __init__(self, input_op, clean_image, is_training=True, num_channels=3,
                 gp_lambda=0.25, learning_rate=2e-4):
        with tf.variable_scope(self.name):
            self.generator = build_unet_generator(input_op, is_training=is_training, num_channels=num_channels)
            self.pi_logits, self.pi_probs = build_unet_policy(input_op, is_training=is_training)
            self.pi_mask = tf.cast(self.pi_probs < 0.5, dtype=tf.float32, name='mask')
            self.d_real, real_ends = build_dcgan_discriminator(clean_image, is_training)
            self.d_fake, fake_ends = build_dcgan_discriminator(self.generator, is_training, reuse=True)

        # Definition of loss function, by default we use WGAN loss with GP
        d_real_loss = tf.reduce_mean(tf.negative(self.d_real))
        d_fake_loss = tf.reduce_mean(self.d_fake)
        self.g_loss = tf.negative(d_fake_loss)

        # Gradient penalty
        epsilon = tf.random_uniform(shape=self.generator.get_shape(), minval=0.0, maxval=1.0)
        interpolates = self.generator + epsilon * (self.generator - clean_image)
        d_gp, _ = build_dcgan_discriminator(interpolates, is_training, reuse=True)
        grads = tf.gradients(d_gp, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        gp = tf.reduce_mean(tf.square(slopes - 1), name='gradient_penalty')
        self.d_loss = d_real_loss + d_fake_loss + gp_lambda * gp

        # Reconstruction loss (termed as valid loss in partial-conv paper)
        masked = (self.generator - input_op) * tf.stop_gradient(self.pi_mask)
        masked = tf.reshape(masked, [args.batch_size, -1])
        self.rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(masked), axis=-1))

        # Total-Variation loss
        self.tv_loss = total_variation_loss(self.generator)

        # Perceptual loss
        alpha = args.alpha
        perceptual_loss = 0
        for key in real_ends.keys():
            # different with partial conv paper, we use alpha with exponential decay
            perceptual_loss += alpha * tf.reduce_mean(tf.abs(real_ends[key] - fake_ends[key]))
            alpha *= 0.5
        self.perceptual_loss = perceptual_loss
        


def pretrain():
    pass


def train():
    pass


def evaluate():
    pass


if __name__ == '__main__':
    if args.function == 'train':
        train()
    elif args.function == 'eval':
        evaluate()
    elif args.function == 'pretrain':
        pretrain()
    else:
        raise NotImplementedError('Invalid function input')