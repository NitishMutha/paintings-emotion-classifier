# Group
# Model: Baseline Convolutional Neural Network
# Purpose: Function file for storage of misccelanous functions for Baseline CNN
# Developers: Russel Daries, Lewis Moffat, Rafiel Faruq, Hugo Phillion, Nitish Mutha

import tensorflow as tf
import numpy as np
import os,sys
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.framework import dtypes
from sklearn.metrics import confusion_matrix

# Weight function creation
def weight_variables(shape):
  ini_w = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(ini_w)

# Bias variable function
def bias_variables(shape):
  ini_b = tf.constant(0.1, shape=shape)
  return tf.Variable(ini_b)

# 2D convolution function
def conv_2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max-Pooling function
def maxpool_2by2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# Function to create 1-hot encoding vectors for emotion classes
def one_hot():

    emotion_range = np.arange(0,5)
    one_hot_vec = np.zeros((5,5),dtype='float32')
    one_hot_vec[np.arange(5),emotion_range] = 1
    return one_hot_vec

# Function to read in input images and greyscale, resize, normalize and pack into nessccary data structure
def resize_images(image_directory,emotions,image_dimension):

    folder = ['train','test']
    emotion_label_count = 0
    emo_count = 0
    emo_label_vec = one_hot()
    # Loop through emotions
    for emotion in emotions:

        # Create one-hot vector for labels for each emotion
        class_label = emo_label_vec[emo_count,:]
        # Loop through folders
        for file in folder:
            emotion_images = []
            emotion_labels = []
            crop_directory = image_directory+ '/'+emotion+'/crop/'+file+'/'

            path = crop_directory
            dirs = os.listdir(path)
            # Go through each image
            for item in dirs:

                imgExts = ["jpeg"]
                ext = item[-4:].lower()
                if ext not in imgExts:
                    continue
                im = Image.open(path+item).convert('L')
                f, e = os.path.splitext(path+item)
                imResize = im.resize((image_dimension,image_dimension), Image.ANTIALIAS)
                image_temp = np.asarray(imResize)
                emotion_images.append(image_temp)
                emotion_labels.append(class_label)
                imResize.save(f + '_resized.jpg','JPEG', quality=100)

            # Call import_dataset class to sort images and labels
            emotion_images = np.asarray(emotion_images)
            emotion_labels = np.asarray(emotion_labels)

            # Sort through each emotion
            if (emo_count==0):
                if(file=='train'):
                    anger_train = import_dataset(emotion_images, emotion_labels)
                else:
                    anger_test = import_dataset(emotion_images, emotion_labels)

            elif (emo_count==1):
                if (file == 'train'):
                    happy_train = import_dataset(emotion_images, emotion_labels)
                else:
                    happy_test = import_dataset(emotion_images, emotion_labels)

            elif (emo_count == 2):
                if (file == 'train'):
                    fear_train = import_dataset(emotion_images, emotion_labels)
                else:
                    fear_test = import_dataset(emotion_images, emotion_labels)
            elif (emo_count == 3):
                if (file == 'train'):
                    neutral_train = import_dataset(emotion_images, emotion_labels)
                else:
                    neutral_test = import_dataset(emotion_images, emotion_labels)
            elif (emo_count == 4):
                if (file == 'train'):
                    sad_train = import_dataset(emotion_images, emotion_labels)
                else:
                    sad_test = import_dataset(emotion_images, emotion_labels)
            else:
                print('Warning: Images not assigned.')

        emo_count += 1

    # Pack all emotions into one complete data structure
    dataset_train = combine_images(anger_train,  happy_train, fear_train,  neutral_train,
                        sad_train)

    dataset_test = combine_images(anger_test,happy_test,fear_test,
                                   neutral_test, sad_test)

    # Return combined training and test set for use in CNNs
    return dataset_train,dataset_test


class import_dataset(object):
    # class to create instance of data which has functions for manipulation such as batch sampling
    def __init__(self,images,labels,dtype = dtypes.float32,reshape=True):

        self._num_examples = images.shape[0]
        if reshape:
            # assert images.shape[3] == 1
            images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        # self._epochs_completed = 0
        # self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

class combine_images(object):

    def __init__(self,anger_t,happy_t,fear_t,neutral_t,sad_t):

        _images_full_t = anger_t.images
        _images_full_t = np.concatenate((_images_full_t, happy_t.images), axis=0)
        _images_full_t = np.concatenate((_images_full_t, fear_t.images), axis=0)
        _images_full_t = np.concatenate((_images_full_t, neutral_t.images), axis=0)
        _images_full_t = np.concatenate((_images_full_t, sad_t.images), axis=0)

        _labels_full_t = anger_t.labels
        _labels_full_t = np.concatenate((_labels_full_t, happy_t.labels), axis=0)
        _labels_full_t = np.concatenate((_labels_full_t, fear_t.labels), axis=0)
        _labels_full_t = np.concatenate((_labels_full_t, neutral_t.labels), axis=0)
        _labels_full_t = np.concatenate((_labels_full_t, sad_t.labels), axis=0)

        self._images = _images_full_t
        self._labels = _labels_full_t

        self._num_examples = _images_full_t.shape[0]

        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:

            # Finished epoch
            self._epochs_completed += 1
            # # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    # with tf.variable_scope(scope):

    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                 name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                  name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])

        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    print('313131')
    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))

    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed

# Function to plot the calculated cross-entropies and accuracy
def plot_image_metrics(metric,x_label,y_label,filename,colour):

    epochs_numbers = metric.shape[0]
    epoch_vec = np.arange(epochs_numbers)
    epoch_vec = epoch_vec+1

    plt.plot(epoch_vec,metric,colour+'o-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(filename + '_metrics.pdf', bbox_inches='tight', format='pdf')
    plt.close()

# Function for creating confusion matrix
def confusion_matrix_plot(y_true_labels, y_p, name, norm = False):

    matrix = confusion_matrix(y_true_labels, y_p)

    if norm ==True:
        matrix_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    else:
        matrix_norm = matrix

    plt.matshow(matrix_norm)
    plt.colorbar()
    plt.ylabel('Predicated Label')
    plt.xlabel('True Label')
    plt.savefig(name + '_metrics.pdf', bbox_inches='tight', format='pdf')
