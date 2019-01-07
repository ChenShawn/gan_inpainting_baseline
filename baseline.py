import tensorflow as tf
import numpy as np
import argparse

from gan_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='celeba', help='support only CelebA currently')
parser.add_argument('-f', '--function', type=str, default='train', help='`pretrain`, `train` or `eval`')
parser.add_argument('-l', '--logdir', type=str, default='./logs', help='dir for tensorboard logs')
parser.add_argument('-s', '--save_dir', type=str, default='./ckpt', help='dir for checkpoints')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-i', '--iterations', type=int, default=20000, help='number of training iterations')
parser.add_argument('-p', '--pretrain', type=bool, default=True, help='whether to load model from ckpt')
parser.add_argument('-a', '--allow_growth', type=bool, default=True, help='whether to grab all gpu resources')
args = parser.parse_args()


class BaselineModel(object):
    def __init__(self, input_op):
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
    else:
        raise NotImplementedError('Invalid function input')

