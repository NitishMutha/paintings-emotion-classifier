# Group
# Model: Baseline Convolutional Neural Network
# Purpose: Baseline CNN for classifying emotions in images
# Developers: Russel Daries, Lewis Moffat, Rafiel Faruq, Hugo Phillion, Nitish Mutha

# Add additional directories
import sys
# Directory for common function files
sys.path.insert(0, '../../common')

# Nesscary Imports
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from PIL import Image
from misc_functions import *
from cnn_functions import *

#--------- Section for importing image dataset ----------#
# Read data from file or function and re-size
image_directory = '../../data'
emotions = ['anger','happy','fear','neutral','sad']
# emotions = ['anger']
image_dimension = 200

dataset_train,dataset_test = resize_images(image_directory,emotions,image_dimension)

#--------- Section for importing image dataset ----------#

# Tensorboard training directory
logs_path = '../model'
print('Log Path: '+ logs_path)

# Model Saving directory
model_path = '../model/tf_baseline_cnn.ckpt'
print('Model Path: ' + model_path)

# Plot naming
# Accuracy
x_label_acc = 'Epoch'
y_label_acc = 'Accuracy'
filename_acc = '../../results/baseline_acc'
# Loss
x_label_loss = 'Epoch'
y_label_loss = 'Cross-Entropy Loss'
filename_loss = '../../results/baseline_loss'

start_time = time.ctime()

# Boolean statements
TRAINING_MODE = True
PLOT_MODE = True
SAVE_MODE = True
ENABLE_TENSORBOARD = True
TEST_MODEL = True

# Size Declarations
OPTIMIZER = 'Adam'
EPOCHS = 1
batch_size_train = 100
batch_size_test = 100

image_dimension_sq = image_dimension * image_dimension
learning_rate = 0.001
emotion_classes = 5
display_step = 5

# Defining initial place holder variables
x = tf.placeholder(tf.float32, shape=[None, image_dimension_sq])
y_ = tf.placeholder(tf.float32, shape=[None, emotion_classes])
keep_prob = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool, name='phase_train')

# # Create Convolutional Neural-Network
layer_1 = ConvPoolLayer(x,5,32,1,image_dimension,'relu',keep_prob,phase_train,
                        True,False,True,False)
layer_2 = ConvPoolLayer(layer_1.output,5,64, 32,image_dimension,'relu',
                        keep_prob,phase_train,False,False,True,False)
layer_3 = DenselyConnectedLayer(layer_2.output,50,64,1024,'relu',keep_prob,False,True)
layer_4 = ReadOutLayer(layer_3.output,1024,emotion_classes,keep_prob,True)

y = layer_4.output

# Cross-entropy calculation
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))

# Optimizer selection
if(OPTIMIZER=='Adam'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
elif(OPTIMIZER=='SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
elif(OPTIMIZER=='Momentum'):
    optimizer = tf.train.MomentumOptimizer(learning_rate).minimize(cost)
else:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Create a summary to monitor cost tensor
tf.summary.scalar("Loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("Accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # Condtional statement to restore the model or start training the model
    if(TRAINING_MODE):

        #Write logs to Tensorboard directory
        if(ENABLE_TENSORBOARD):
            print('Tensorboard Enabled.')
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        else:
            print('Tensorboard Disabled')

        print('----Training Baseline CNN Model Running----')
        # Keep training until reach max iterations
        train_acc = 0.0
        train_loss = 0.0
        train_acc_vec = []
        train_loss_vec = []

        # Number of loop repititions of model
        for epoch in range(EPOCHS):

            avg_loss = 0.0
            avg_acc = 0.0

            total_batch_train = int(dataset_train.num_examples/batch_size_train)

            # Loop for number of batches for training data
            for j in range(total_batch_train):

                batch_x_train, batch_y_train = dataset_train.next_batch(batch_size_train)

                # Run optimization
                sess.run(optimizer, feed_dict={x: batch_x_train, y_: batch_y_train, keep_prob: 0.5, phase_train: True})
                acc, loss, summary = sess.run([accuracy, cost, merged_summary_op],
                                              feed_dict={x: batch_x_train, y_: batch_y_train, keep_prob: 1.0, phase_train: False})

                if(ENABLE_TENSORBOARD):
                    summary_writer.add_summary(summary, epoch * total_batch_train + j)
                else:
                    pass
                avg_acc += acc
                avg_loss += loss

                # Print condition to check state of model
                if j % display_step == 0:
                    print("Epoch " + str(epoch + 1) + ", Batch Number " + str(j) + ", Batch Loss= " + \
                          "{:.5f}".format(loss) + ", Batch Accuracy= " + \
                          "{:.5f}".format(acc))

            # Average out accuracy over batches
            avg_acc = avg_acc / total_batch_train
            avg_loss = avg_loss / total_batch_train
            train_acc+=avg_acc
            train_loss+=avg_loss
            train_acc_vec.append(avg_acc)
            train_loss_vec.append(avg_loss)

            print("Epoch " + str(epoch + 1) + ", Epoch Loss= " + "{:.5f}".format(avg_loss) + ", Epoch Accuracy= " + \
                  "{:.5f}".format(avg_acc))

        print("Optimization Finished!")

        train_acc_vec = np.asarray(train_acc_vec)
        train_loss_vec = np.asarray(train_loss_vec)

        if(PLOT_MODE):
            print('Plot Mode enabled.')
            plot_image_metrics(train_acc_vec, x_label_acc, y_label_acc, filename_acc)
            plot_image_metrics(train_loss_vec, x_label_loss, y_label_loss, filename_loss)
        else:
            print('Plot Mode disabled.')

        train_loss = train_loss/epoch
        train_acc = train_acc/epoch

        end_time = time.ctime()
        # Check the training time it took the model to complete.
        print("Training Start Time : ", start_time)
        print("Training End Time : ", end_time)

        if(SAVE_MODE):
            print('Save mode enabled.')
            # Saving the model
            save_path = saver.save(sess, model_path)
            print('Model saved to file: ', model_path)
        else:
            print('Save mode disabled.')
            print('Model not saved.')

        # Display training accuracy and loss achieved.
        print("Train Loss: "+ "{:.5f}".format(train_loss))
        print("Train Accuracy: "+ "{:.5f}".format(train_acc))

        if(TEST_MODEL):

            print('Testing Model.')

            avg_loss_test = 0.0
            avg_acc_test = 0.0

            total_batch_test = int(dataset_test.num_examples / batch_size_test)

            # Loop for number of batches for training data
            for k in range(total_batch_test):

                batch_x_test, batch_y_test = dataset_test.next_batch(batch_size_test)

                # Run optimization
                acc, loss, summary = sess.run([accuracy, cost, merged_summary_op],
                                                        feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.0,
                                                                   phase_train: False})

                avg_acc_test += acc
                avg_loss_test += loss

            # Average out accuracy over batches
            test_acc = avg_acc_test / total_batch_test
            test_loss = avg_loss_test / total_batch_test

            # Display training accuracy and loss achieved.
            print("Test Loss: " + "{:.5f}".format(test_loss))
            print("Test Accuracy: " + "{:.5f}".format(test_acc))

        else:
            print('Not testing model.')
    else:

        # Restore the model for testing purposes
        print('----Test Mode Running----')
        load_path = saver.restore(sess, model_path)
        print("Baseline Model restored from file: ", model_path)

        avg_loss_test = 0.0
        avg_acc_test = 0.0

        total_batch_test = int(dataset_test.num_examples / batch_size_test)

        # Loop for number of batches for training data
        for k in range(total_batch_test):
            batch_x_test, batch_y_test = dataset_test.next_batch(batch_size_test)

            # Run optimization
            acc, loss, summary = sess.run([accuracy, cost, merged_summary_op],
                                          feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.0,
                                                     phase_train: False})

            avg_acc_test += acc
            avg_loss_test += loss

        # Average out accuracy over batches
        test_acc = avg_acc_test / total_batch_test
        test_loss = avg_loss_test / total_batch_test

        # Display training accuracy and loss achieved.
        print("Test Loss: " + "{:.5f}".format(test_loss))
        print("Test Accuracy: " + "{:.5f}".format(test_acc))




