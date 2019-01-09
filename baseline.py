import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm

from gan_utils import *
from utils import show_all_variables, total_variation_loss, save, load
from data import CelebAReader

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='celeba', help='support only CelebA currently')
parser.add_argument('-f', '--function', type=str, default='train', help='`pretrain`, `train` or `eval`')
parser.add_argument('--logdir', type=str, default='./logs', help='dir for tensorboard logs')
parser.add_argument('--model_path', type=str, default='./ckpt', help='dir for checkpoints')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-i', '--iterations', type=int, default=20000, help='number of training iterations')
parser.add_argument('-p', '--pretrain', type=bool, default=True, help='whether to load model from ckpt')
parser.add_argument('-a', '--allow_growth', type=bool, default=True, help='whether to grab all gpu resources')
parser.add_argument('--gp_lambda', type=float, default=0.25, help='coefficient for GP loss term')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning_rate')
parser.add_argument('--alpha', type=float, default=0.05, help='initial value of perceptual loss coefficient')
parser.add_argument('--beta', type=float, default=0.1, help='coefficient for TV loss')
parser.add_argument('--write_logs_every', type=int, default=100, help='write_logs_every')
parser.add_argument('--save_images_every', type=int, default=500, help='save_images_every')
parser.add_argument('--critic_iter', type=int, default=2, help='number of iterations to update critics')
args = parser.parse_args()


class BaselineModel(object):
    name = 'BaselineModel'

    def __init__(self, input_op, clean_image, mask, is_training=True, num_channels=3):
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
        self.d_loss = d_real_loss + d_fake_loss + args.gp_lambda * gp

        # Reconstruction loss (termed as valid loss in partial-conv paper)
        masked = (self.generator - input_op) * self.pi_mask
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
        # Policy loss used for pretrain
        self.pi_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pi_logits,
                                                                              labels=mask))
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
            tf.summary.image('predicted_mask', self.pi_probs),
            tf.summary.image('ground_truth_mask', mask),
            tf.summary.image('generator', self.generator)
        ], name='img_sums')
        self.pi_sum = tf.summary.scalar('pi_loss', self.pi_loss)
        print(' [*] Summaries built finished')

        # Definition for optimizers
        train_vars = tf.trainable_variables()
        pi_vars = [var for var in train_vars if 'Policy' in var.name]
        g_vars = [var for var in train_vars if 'Generator' in var.name]
        d_vars = [var for var in train_vars if 'Discriminator' in var.name]
        self.pi_optim = tf.train.AdamOptimizer(args.learning_rate).minimize(self.pi_loss, var_list=pi_vars)
        self.g_optim = tf.train.AdamOptimizer(args.learning_rate, 0.5, 0.9).minimize(self.g_loss, var_list=g_vars)
        self.d_optim = tf.train.AdamOptimizer(args.learning_rate, 0.5, 0.9).minimize(self.d_loss, var_list=d_vars)
        print(' [*] Optimizers definition finished')



def pretrain(sess, model, global_step=0):
    counter = global_step
    writer = tf.summary.FileWriter(args.logdir, sess.graph)
    for iter in range(args.iterations):
        try:
            sess.run(model.pi_optim)
            # Write logs for tensorboard visualization
            if iter % args.write_logs_every:
                sum_str = sess.run(model.pi_sum)
                writer.add_summary(sum_str)
            if iter % args.save_images_every:
                sum_str = sess.run(model.img_sum)
                writer.add_summary(sum_str)
        except tf.errors.OutOfRangeError:
            break
        else:
            counter += 1
    print(' [*] Training finished, ready to save...')
    saved_path = save(sess, args.model_path, model_name=model.name + '.model', global_step=counter)
    print(' [*] Successfully save the model in ' + saved_path)
    return counter


def train(sess, model, global_step=0):
    counter = global_step
    writer = tf.summary.FileWriter(args.logdir, sess.graph)
    for iter in range(args.iterations):
        try:
            for jt in range(args.critic_iter):
                sess.run(model.d_optim)
            sess.run(model.g_optim)

            # Write logs for tensorboard visualization
            if iter % args.write_logs_every == 1:
                g_sum_str, d_sum_str = sess.run([model.g_sum, model.d_sum])
                writer.add_summary(g_sum_str, counter)
                writer.add_summary(d_sum_str, counter)
            if iter % args.save_images_every == 1:
                img_sum_str = sess.run(model.img_sum)
                writer.add_summary(img_sum_str)
            # Update global_step
            counter += 1
        except tf.errors.OutOfRangeError:
            break
    print(' [*] Training finished, ready to save...')
    saved_path = save(sess, args.model_path, model_name=model.name + '.model', global_step=counter)
    print(' [*] Successfully save the model in ' + saved_path)
    return counter


def evaluate():
    pass


if __name__ == '__main__':
    if args.function == 'train':
        executed = train
    elif args.function == 'eval':
        executed = evaluate
    elif args.function == 'pretrain':
        executed = pretrain
    else:
        raise NotImplementedError('Invalid function input')

    if args.dataset == 'celeba':
        dataset = CelebAReader()

    ans = executed()