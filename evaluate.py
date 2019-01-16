import tensorflow as tf
from data import MnistReader
from utils import load
from mnist_model import BaselineModel

if __name__=='__main__':
    reader = MnistReader(size=(28, 28), type='x_test')

    # load model
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        model = BaselineModel(reader.lossy_xs, reader.batch_xs, reader.mask, num_channels=1)
        status, global_step= load(sess, './ckpt')
        count = global_step
        writer = tf.summary.FileWriter('./logs', sess.graph)
        for iter in range(10000):
            try:
                writer
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.InvalidArgumentError:
                continue

