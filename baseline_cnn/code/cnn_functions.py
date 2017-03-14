# Group
# Purpose: Function file for storage of CNN layer classes

import tensorflow as tf
import numpy as np
import pandasas pd
import math
from misc_functions import *


class ConvPoolLayer(object):

    def __init__(self,input,patch_size,features,input_channels,image_dimension,activation,prob,input_layer=False,batch_norm=True,max_pool=True,dropout=True,training_phase=True):

        self.input = input
        self.W = weight_variables([patch_size,patch_size,input_channels,features])
        self.b = bias_variables([features])

        if(input_layer):
            temp_calc = tf.reshape(input,[-1,image_dimension,image_dimension,1])
        else:
            temp_calc = self.input

        # Layer Batch Normalisation
        if(batch_norm):

            temp = conv2d(temp_calc, self.W) + self.b

            self.z = batch_norm(temp, input_channels, training_phase, scope='bn')

        else:
            self.z = conv2d(temp_calc, self.W) + self.b

        # Layer Activation Function
        if(activation=='relu')
            self.h_conv = tf.nn.relu(self.z)
        else:
            print('!!!!Unknown activation function specified!!!!!!')
            pass

        # Layer Max-Pooling
        if(max_pool):
            self.h_pool = max_pool_2x2(self.h_conv)
        else:
            self.h_pool = self.h_conv

        # Layer Dropout
        if(dropout):
            self.h_pool_drop = tf.nn.dropout(self.h_pool,prob)
        else:
            self.h_pool_drop = self.h_pool

        self.output = self.h_pool_drop


class DenselyConnectLayer(object):

    def __init__(self,input,reduced_patch_size,features,output_neurons,activation,prob,batch_norm=True,dropout=True):


        self.input = input
        self.W = weight_variables([reduced_patch_size*reduced_patch_size*features,output_neurons])
        self.b = bias_variables([output_neurons])

        self.h_pool_flat = tf.reshape(input,[-1,reduced_patch_size*reduced_patch_size*features])

        # Batch Normalization
        if(batch_norm):

            batch_mean, batch_var = tf.nn.moments(self.h_pool_flat, [0])

            scale = tf.Variable(tf.ones([output_neurons]))

            beta = tf.Variable(tf.zeros([output_neurons]))

            self.h_pool_flat_bn = tf.nn.batch_normalization(self.h_pool_flat, batch_mean, batch_var, beta, scale, 1e-3)

        else:
            self.h_pool_flat_bn = self.h_pool_flat


        # Activation Function
        if(activation=='relu')
            self.h_fc = tf.nn.relu(tf.matmul(self.h_pool_flat_bn , self.W) + self.b)
        else:
            pass

        # Dropout Application
        if(dropout):
            self.h_fc_drop = tf.nn.dropout(self.h_fc,prob)
        else:
            self.h_fc_drop = self.h_fc

        self.output = self.h_fc_drop


class ReadOutLayer(object):

    def __init__(self,input,input_neurons,emotion_classes,prob,dropout):

        self.W = weight_variables([input_neurons,emotion_classes])
        self.b = bias_variables([emotion_classes])

        if(dropout):
            self.input = tf.nn.dropout(input,prob)
        else:
            self.input = input

        self.y = tf.matmul(self.input, self.W) + self.b

