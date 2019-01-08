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
parser.add_argument('--beta', type=float, default=0.1, help='coefficient for TV loss')
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
        print(' [*] Build model finished')

        # Definition of loss function, by default we use WGAN loss with GP
        d_real_loss = tf.reduce_mean(tf.negative(self.d_real))
        d_fake_loss = tf.reduce_mean(self.d_fake)
        g_loss = tf.negative(d_fake_loss)

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
        alpha = 1.0
        perceptual_loss = 0
        for key in real_ends.keys():
            # different with partial conv paper, we use alpha with exponential decay
            perceptual_loss += alpha * tf.reduce_mean(tf.abs(real_ends[key] - fake_ends[key]))
            alpha *= 0.5
        self.perceptual_loss = perceptual_loss

        # These magic coefficients are also given by the partial conv paper
        self.g_loss = g_loss + self.rec_loss + args.beta * self.tv_loss + args.alpha * self.perceptual_loss
        print(' [*] Loss function definition finished')

        # Definition for the summaries
        self.g_sum = tf.summary.merge([
            tf.summary.scalar('g_loss', g_loss),
            tf.summary.scalar('rec_loss', self.rec_loss),
            tf.summary.scalar('tv_loss', self.tv_loss),
            tf.summary.scalar('perceptual_loss', self.perceptual_loss),
            tf.summary.scalar('g_total', self.g_loss)
        ], name='gen_sums')
        self.d_sum = tf.summary.merge([
            tf.summary.scalar('d_real', d_real_loss),
            tf.summary.scalar('d_fake', d_fake_loss),
            tf.summary.scalar('gp', gp),
            tf.summary.scalar('d_total', self.d_loss),
            tf.summary.histogram('d_real_hist', self.d_real),
            tf.summary.histogram('d_fake_hist', self.d_fake)
        ], name='disc_sums')
        self.img_sum = tf.summary.merge([
            tf.summary.image('input_op', input_op),
            tf.summary.image('clean_image', clean_image),
            tf.summary.image('mask', self.pi_probs),
            tf.summary.image('generator', self.generator)
        ])
        print(' [*] Summaries built finished')

        # Definition for optimizers
        train_vars = tf.trainable_variables()
        g_vars = [var for var in train_vars if 'Generator' in var.name]
        d_vars = [var for var in train_vars if 'Discriminator' in var.name]
        self.g_optim = tf.train.AdamOptimizer(args.learning_rate, 0.5, 0.9).minimize(self.g_loss, var_list=g_vars)
        self.d_optim = tf.train.AdamOptimizer(args.learning_rate, 0.5, 0.9).minimize(self.d_loss, var_list=d_vars)
        print(' [*] Optimizers definition finished')



def pretrain():
    pass


def train(sess, model, feed_dict=None):
    for iter in range(args.iterations):
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