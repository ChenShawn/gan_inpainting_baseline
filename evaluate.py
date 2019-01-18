import tensorflow as tf
from data import MnistReader
from utils import load
from mnist_model import BaselineModel
import numpy as np
import cv2, os

# arr should be 4 dimensional
def save_image(arr, name, idx, scale=True, path='D:\\Inpainting\\generated\\'):
    # clean files
    if scale:
        arr = arr * (255.0 / np.max(arr))
    for i in range(arr.shape[0]):
        img_to_save = arr[i, :, :, :].astype(np.uint8)
        cv2.imwrite(path + str(idx) + '_' + str(i) + '_' + name, img_to_save)
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



if __name__=='__main__':
    gen_saved_path = 'D:\\Inpainting\\generated'
    masked_saved_path = 'D:\\Inpainting\\inputs'
    reader = MnistReader(size=(28, 28), batch_size=64, type='test')

    # load model
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        model = BaselineModel(reader.lossy_xs, reader.batch_xs, reader.mask, num_channels=1, is_training=False)
        load(sess, './ckpt')
        for iter in range(100):
            gen_images, masked_images = collect_images(sess, model, reader)
            gen_images = (255.0 * gen_images).astype(np.uint8)
            cv2.imwrite(os.path.join(gen_saved_path, 'gen_' + str(iter) + '.png'), gen_images)
            cv2.imwrite(os.path.join(masked_saved_path, 'masked_' + str(iter) + '.png'), masked_images)
    print(' [*] Testing finished, ready to save...')