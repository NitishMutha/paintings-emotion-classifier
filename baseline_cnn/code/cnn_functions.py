# Group
# Purpose: Function file for storage of CNN layer classes

import tensorflow as tf
import numpy as np

from misc_functions import *


class ConvPoolLayer(object):

    def __init__(self,input,patch_size,features,input_channels,image_dimension,activation,INPUT_LAYER):

        self.input = input
        self.W = weight_variables([patch_size,patch_size,input_channels,features])
        self.b = bias_variables([features])

        if(INPUT_LAYER):
            temp_calc = tf.reshape(input,[-1,image_dimension,image_dimension,1])
        else:
            temp_calc = self.input

        temp = conv2d(temp_calc, self.W) + self.b

        # Activation Function
        if(activation=='relu')
            self.h_conv = tf.nn.relu(temp)
        else:
            print('!!!!Unknown activation function specified!!!!!!')
            pass

        self.h_pool = max_pool_2x2(self.h_conv)

        self.output = self.h_pool


class DenselyConnectLayer(object):

    def __init__(self,input,reduced_patch_size,features,output_neurons,activation):


        self.input = input
        self.W = weight_variables([reduced_patch_size*reduced_patch_size*features,output_neurons])
        self.b = bias_variables([output_neurons])

        self.h_pool_flat = tf.reshape(input,[-1,reduced_patch_size*reduced_patch_size*features])

        # Activation Function
        if(activation=='relu')
            self.h_fc = tf.nn.relu(tf.matmul(self.h_pool_flat , self.W) + self.b)
        else:
            pass

        self.output = self.h_fc


class ReadOutLayer(object):

    def __init__(self,input,input_neurons,emotion_classes,prob,DROPOUT):

        self.W = weight_variables([input_neurons,emotion_classes])
        self.b = bias_variables([emotion_classes])

        if(DROPOUT):
            self.input = tf.nn.dropout(input,prob)
        else:
            self.input = input

        self.y = tf.matmul(self.input, self.W) + self.b

