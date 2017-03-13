# Group
# Purpose: Function file for storage of misccelanous functions for Baseline CNN

import tensorflow as tf
import numpy as np

def weight_variables(shape):
  ini_w = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(ini_w)

def bias_variables(shape):
  ini_b = tf.constant(0.1, shape=shape)
  return tf.Variable(ini_b)

def conv_2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool_2by2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def split_data(data):
    # Function to split input data in training and test set

    return dataset_train,dataset_test


class import_dataset(object):
    # class to create instance of data which has functions for manipulation such as batch sampling
    def __init__(self):

    def num_examples(self):
        return self._num_examples


    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
                ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


        return batch_x,batch_y
