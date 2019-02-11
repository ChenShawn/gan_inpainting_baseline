import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime

from gan_utils import *
from utils import show_all_variables, build_total_variation_loss, build_style_loss, save, load
from data import PlaceReader

"""
    This program is provided by ChenShawn in Jan 17th, 2019 to re-implement the paper
    "Image inpainting for irregular holes using partial convolution"
    
    A brief summary about the paper and notice on this codes:
    1. The paper replaces all the conv layers with partial conv layers
    2. The paper uses a pretrained VGG-16 model, which is a bit tricky to implement the partial conv
    3. The paper has a specifically-designed style loss. Its form may seem a little bit wired.
       It is ambiguous whether the style loss should use all the layers as feature maps,
       or only use pool1, pool2, pool3 as feature maps
    4. The paper start the training with moving bn and learning rate of 2e-4, 
       and finetune the model with fixed bn and a learning rate of 5e-5
    5. Though the paper claims to give a comparison between PConv and Conv (see page 10),
       Neither the Conv results nor the parameters of the Conv model are given.
"""

# TODO: check https://github.com/naoto0804/pytorch-inpainting-with-partial-conv for more info

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='celeba', help='support only CelebA currently')
parser.add_argument('-f', '--function', type=str, default='train', help='`pretrain`, `train` or `eval`')
parser.add_argument('--logdir', type=str, default='./logs/pconv/', help='dir for tensorboard logs')
parser.add_argument('--model_path', type=str, default='./ckpt/pconv/', help='dir for checkpoints')
parser.add_argument('-b', '--batch_size', type=int, default=6, help='batch size')
parser.add_argument('-i', '--iterations', type=int, default=60000, help='number of training iterations')
parser.add_argument('-p', '--pretrain', type=bool, default=True, help='whether to load model from ckpt')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning_rate')
parser.add_argument('--write_logs_every', type=int, default=20, help='write_logs_every')
parser.add_argument('--save_images_every', type=int, default=500, help='save_images_every')
parser.add_argument('--critic_iter', type=int, default=3, help='number of iterations to update critics')
args = parser.parse_args()


class PartialConvModel(object):
    name = 'PConvModel'

    def __init__(self, input_op, clean_image, mask, is_training=True, num_channels=3):
        self.generator, end_points = build_vgg_pretraned_model(input_op, is_training=is_training)
        _, end_points_gt = build_vgg_pretraned_model(clean_image, is_training=False, reuse=True)
        self.comp_image = self.generator * (1.0 - mask) + clean_image * mask
        self.phi_comp, end_points_comp = build_vgg_pretraned_model(self.comp_image, is_training=True, reuse=True)
        print(' [*] Build model finished')

        # Stop gradient to ground_truth in graph, i.e. ground truth should be fixed instead of optimized
        with tf.name_scope('stop_gradient_gt'):
            for key in end_points_gt.keys():
                end_points_gt[key] = tf.stop_gradient(end_points_gt[key])

        # Hole loss and valid loss are processed separately
        mse = tf.square(self.generator - clean_image)
        hole_loss = tf.reduce_mean(tf.reduce_sum(tf.reshape(mask * mse, [args.batch_size, -1]), axis=-1))
        valid_loss = tf.reduce_mean(tf.reduce_sum(tf.reshape((1.0 - mask) * mse, [args.batch_size, -1]), axis=-1))
        self.rec_loss = 0.01 * (6.0 * hole_loss + valid_loss)

        # Total-Variation loss 0.1
        self.tv_loss = 0.5 * build_total_variation_loss(self.generator)

        # Perceptual loss
        perceptual_loss = tf.reduce_mean(tf.abs(end_points['vgg_16/pool1'] - end_points_gt['vgg_16_1/pool1'])) + \
            tf.reduce_mean(tf.abs(end_points['vgg_16/pool2'] - end_points_gt['vgg_16_1/pool2'])) + \
            tf.reduce_mean(tf.abs(end_points['vgg_16/pool3'] - end_points_gt['vgg_16_1/pool3'])) + \
            tf.reduce_mean(tf.abs(end_points_gt['vgg_16_1/pool1'] - end_points_comp['vgg_16_2/pool1'])) + \
            tf.reduce_mean(tf.abs(end_points_gt['vgg_16_1/pool2'] - end_points_comp['vgg_16_2/pool2'])) + \
            tf.reduce_mean(tf.abs(end_points_gt['vgg_16_1/pool3'] - end_points_comp['vgg_16_2/pool3']))
        self.perceptual_loss = 0.05 * perceptual_loss

        # Style loss definition
        style_loss = build_style_loss(end_points['vgg_16/pool1'], end_points_gt['vgg_16_1/pool1']) + \
            build_style_loss(end_points['vgg_16/pool2'], end_points_gt['vgg_16_1/pool2']) + \
            build_style_loss(end_points['vgg_16/pool3'], end_points_gt['vgg_16_1/pool3']) + \
            build_style_loss(end_points_gt['vgg_16_1/pool1'], end_points_comp['vgg_16_2/pool1']) + \
            build_style_loss(end_points_gt['vgg_16_1/pool2'], end_points_comp['vgg_16_2/pool2']) + \
            build_style_loss(end_points_gt['vgg_16_1/pool3'], end_points_comp['vgg_16_2/pool3'])
        self.style_loss = 120.0 * style_loss

        # Policy loss used for pretrain
        self.loss = self.style_loss + self.rec_loss + self.perceptual_loss + self.tv_loss
        print(' [*] Loss function definition finished')

        # Definition for the summaries
        self.scalar_sums = tf.summary.merge([
            tf.summary.scalar('rec_loss', self.rec_loss),
            tf.summary.scalar('perceptual_loss', self.perceptual_loss),
            tf.summary.scalar('tv_loss', self.tv_loss),
            tf.summary.scalar('style_loss', self.style_loss),
            tf.summary.scalar('total_loss', self.loss)
        ], name='scalars')
        self.img_sums = tf.summary.merge([
            tf.summary.image('input_image', input_op),
            tf.summary.image('Generator', self.generator),
            tf.summary.image('ground_truth', clean_image),

            # TODO: remove the following two summaries after debuging

            tf.summary.image('comp_image', self.comp_image),
            tf.summary.image('mask', mask)

        ], name='images')
        print(' [*] Summaries built finished')

        # Definition for optimizers
        train_vars = tf.trainable_variables()
        decoder_vars = [var for var in train_vars if 'Decoder' in var.name]
        self.optim = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        self.pre_optim = tf.train.AdamOptimizer(args.learning_rate).minimize(self.rec_loss, var_list=decoder_vars)
        print(' [*] Optimizers definition finished')

    def load_pretrained_vgg16(self, sess, pretrained_model_dir='./pretrained_model/vgg_16.ckpt'):
        restored_vars = slim.get_variables_to_restore(exclude=['Decoder'])
        init_func = slim.assign_from_checkpoint_fn(pretrained_model_dir, restored_vars, ignore_missing_vars=True)
        init_func(sess)
        print(' [*] Model successfully loaded!')



def finetune(sess, model, global_step=0):
    counter = global_step
    writer = tf.summary.FileWriter(args.logdir, sess.graph)
    # TODO: finish the training codes


def pretrain_decoder(sess, model):
    counter = 0
    print(' [*] Start to pretrain')
    for iter in range(args.iterations):
        try:
            sess.run(model.pre_optim)
            # Write logs on screen
            if iter % args.write_logs_every == 1:
                loss = sess.run(model.rec_loss)
                print(' --Time: {} --Step: {}--Loss: {}'.format(str(datetime.now()), counter, loss))
            counter += 1
        except tf.errors.OutOfRangeError:
            break
        except tf.errors.InvalidArgumentError:
            continue
    print(' [*] Training finished, ready to save...')
    saved_path = save(sess, args.model_path, model_name=model.name + '.model', global_step=1)
    print(' [*] Successfully save the model in ' + saved_path)


def train(sess, model, global_step=0):
    counter = global_step
    writer = tf.summary.FileWriter(args.logdir, sess.graph)
    print(' [*] Start to train from global step', global_step)
    for iter in range(args.iterations):
        try:
            sess.run(model.optim)
            # Write logs for tensorboard visualization
            if iter % args.write_logs_every == 1:
                sumstr = sess.run(model.scalar_sums)
                writer.add_summary(sumstr, counter)
            if iter % args.save_images_every == 1:
                img_sum_str = sess.run(model.img_sums)
                writer.add_summary(img_sum_str, counter)
            # Update global_step
            counter += 1
        except tf.errors.OutOfRangeError:
            break
        except tf.errors.InvalidArgumentError:
            continue
    print(' [*] Training finished, ready to save...')
    saved_path = save(sess, args.model_path, model_name=model.name + '.model', global_step=counter)
    print(' [*] Successfully save the model in ' + saved_path)
    return counter




if __name__ == '__main__':
    if args.dataset == 'places':
        print(' [*] Reading places dataset...')
        # Resize the image to even size to avoid concat-time error
        reader = PlaceReader(size=(224, 224))
    else:
        raise NotImplementedError('Only places supported!')

    # Pretrain model
    if args.function == 'train':
        model = PartialConvModel(reader.lossy_xs, reader.batch_xs, reader.mask)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            model.load_pretrained_vgg16(sess)
            if args.pretrain:
                status, global_step = load(sess, args.model_path)
            else:
                global_step = 0
            show_all_variables()
            # pretrain_decoder(sess, model)
            train(sess, model, global_step=global_step)

    else:
        raise NotImplementedError('Invalid function input')