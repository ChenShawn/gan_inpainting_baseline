import tensorflow as tf
import numpy as np
from collections import namedtuple
import cv2, os
import math

from data import MnistReader,PlaceReader
from utils import load
from mnist_model import BaselineModel
from partial_conv import PartialConvModel



# arr should be 4 dimensional
def save_image(arr, name, idx, scale=True, path='D:\\Inpainting\\generated\\'):
    # clean files
    if scale:
        arr = arr * 255.0
    for i in range(arr.shape[0]):
        img_to_save = arr[i, :, :, :].astype(np.uint8)
        cv2.imwrite(os.path.join(path, str(idx) + '_' + str(i) + '_' + name), img_to_save)
    print('SAVING GENERATED IMAGES TO: ' + path + name)


def collect_images(sess, model, reader, batch_size=64):
    gen = [[] for _ in range(10)]
    masked = [[] for _ in range(10)]
    while not all([len(gen[it]) == 10 for it in range(10)]):
        try:
            imgs, inputs, labels = sess.run([model.generator, reader.lossy_xs, reader.batch_ys])
        except tf.errors.InvalidArgumentError:
            continue
        for it in range(batch_size):
            if len(gen[labels[it]]) < 10:
                gen[labels[it]].append(imgs[it, :, :, 0])
                masked[labels[it]].append(inputs[it, :, :, 0])
    gen_images = np.vstack([np.hstack(arrs) for arrs in gen])
    masked_images = np.vstack([np.hstack(arrs) for arrs in masked])
    return gen_images, masked_images


def evaluate(input_op, clean_imgs, eval_type):
    """
    :param input_op: generated images
    :param clean_imgs: groundtruth images
    :param eval_type: [mse, psnr, ssim]
    :return: the metrics of generated images
    """
    batch_size = input_op.get_shape().as_list()[0]
    input_op = tf.reshape(input_op, [batch_size, -1])
    clean_imgs = tf.reshape(clean_imgs, [batch_size, -1])
    if eval_type == 'mse':
        result = tf.reduce_mean(tf.squared_difference(input_op, clean_imgs), axis=0)
    elif eval_type == 'psnr':
        mse = tf.reduce_mean(tf.squared_difference(input_op, clean_imgs), axis=0)
        psnr = tf.constant(255 ** 2, dtype=tf.float32) / mse
        result = tf.log(psnr)
    elif eval_type == 'ssim':
        result = tf.image.ssim(input_op, clean_imgs)
    else:
        raise NotImplementedError
    return result



class EvaluatorBase(object):
    ConfigProto = namedtuple('ConfigProto', ['dataset', 'metric', 'images_save_dir'])

    def __init__(self, *args, **kwargs):
        self.config = self.ConfigProto(*args, **kwargs)
        print(self.config)


    def build_metric(self, generated, ground_truth, epsilon=1e-25):
        """build_metric
        Only `mse` and `psnr` are supported currently
        :param generated: type tf.Tensor
        :param ground_truth: type tf.Tensor
        """
        if self.config.metric.lower() == 'mse':
            batch_size = generated.get_shape().as_list()[0]
            diff = (255.0 ** 2) * tf.squared_difference(generated, ground_truth)
            diffsum = tf.reduce_sum(tf.reshape(diff, [batch_size, -1]), axis=-1)
            self.metric = tf.reduce_mean(diffsum)

        elif self.config.metric.lower() == 'psnr':
            batch_size = generated.get_shape().as_list()[0]
            diff = (255.0 ** 2) * tf.squared_difference(generated, ground_truth)
            diffsum = tf.reduce_sum(tf.reshape(diff, [batch_size, -1]), axis=-1)
            psnr = 10.0 * (math.log10(255.0) - (tf.log(diffsum + epsilon) / math.log(10.0)))
            self.metric = tf.reduce_mean(psnr)
        else:
            raise NotImplementedError


    def evaluate(self, sess, write_logs_every=200):
        moving_mean = 0.0
        counter = 0.0
        while True:
            counter += 1.0
            try:
                tmpval = sess.run(self.metric)
                moving_mean += (1.0 / counter) * (tmpval - moving_mean)
            except tf.errors.InvalidArgumentError:
                continue
            except tf.errors.OutOfRangeError:
                break
            if int(counter) % write_logs_every == 1:
                print(' --Step {} --Mean {} --TmpVal {}'.format(int(counter), moving_mean, tmpval))
        return moving_mean


    def write_images(self, sess, input_op, generator, batch_ys, num_batch):
        """write_images
        :param sess: type tf.Session
        :param num_batch: type int
        """
        counter = 0
        while counter < num_batch:
            try:
                input_images, generated, ground_truth = sess.run([input_op, generator, batch_ys])
            except tf.errors.InvalidArgumentError:
                continue
            except tf.errors.OutOfRangeError:
                continue
            else:
                counter += 1

            save_image(input_images, 'input', iter, path=os.path.join(self.config.images_save_dir, 'inputs'))
            save_image(generated, 'gen', iter, path=os.path.join(self.config.images_save_dir, 'generated'))
            save_image(ground_truth, 'grt', iter, path=os.path.join(self.config.images_save_dir, 'ground_truth'))
        print('Write images finished!')



class EvaluatorPlaces(EvaluatorBase):
    def __init__(self, metric, images_save_dir='D:/Inpainting/places/'):
        super(EvaluatorPlaces, self).__init__(metric=metric, dataset='places', images_save_dir=images_save_dir)



def test_dataset(dataset, readermodel, basemodel):
    if dataset == 'mnist':
        gen_saved_path = 'D:\\Inpainting\\mnist\\generated'
        masked_saved_path = 'D:\\Inpainting\\mnist\\inputs'
        reader = readermodel(size=(28, 28), batch_size=64, type='test')

        # load model
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            model = basemodel(reader.lossy_xs, reader.batch_xs, reader.mask, num_channels=1, is_training=False)
            load(sess, './ckpt')
            for iter in range(100):
                gen_images, masked_images = collect_images(sess, model, reader)
                gen_images = (255.0 * gen_images).astype(np.uint8)
                cv2.imwrite(os.path.join(gen_saved_path, 'gen_' + str(iter) + '.png'), gen_images)
                cv2.imwrite(os.path.join(masked_saved_path, 'masked_' + str(iter) + '.png'), masked_images)

    elif dataset == 'places':
        gen_saved_path = 'D:\\Inpainting\\places\\generated'
        masked_saved_path = 'D:\\Inpainting\\places\\inputs'
        reader = readermodel(size=(224, 224), batch_size=32, type='test')

        #load model
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            model = basemodel(reader.lossy_xs, reader.batch_xs, reader.mask, is_training=False)
            load(sess, './ckpt/pconv/')
            for iter in range(10):
                count = 0
                imgs, inputs = sess.run([model.generator, reader.lossy_xs])
                imgs = (255.0 * imgs).astype(np.uint8)
                inputs = (255.0 * inputs).astype(np.uint8)
                for j in range(reader.batch_size):
                    cv2.imwrite(os.path.join(gen_saved_path, 'gen_' + str(iter) + '_' + str(j) + '.png'), imgs[j, :, :, :])
                    cv2.imwrite(os.path.join(masked_saved_path, 'masked_' + str(iter) + '_' + str(j) + '.png'), inputs[j, :, :, :])
    print(' [*] Testing finished, ready to save...')



if __name__=='__main__':
    #reader = PlaceReader()
    # model = PartialConvModel(reader.lossy_xs, reader.batch_xs, reader.mask, is_training=False)
    test_dataset('places', PlaceReader, PartialConvModel)