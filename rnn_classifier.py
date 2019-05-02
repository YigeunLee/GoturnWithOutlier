import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
import os
import glob
import sys
import random

n_inputs = 4
n_outputs = 4
n_steps = 1
learning_rate = 0.001
n_iterations = 1500
batch_size = 10


X0 = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
Y0 = tf.placeholder(tf.float32,[None,n_steps,n_outputs])

basic_cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons))
outputs , states = tf.nn.dynamic_rnn(basic_cell,X0,dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - 1))
optimizer = tf.train.AdamOptimzer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initalizer()

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch,Y_batch = get_batch()
        sess.run(trainng_op,feed_dict={X0:x_batch,Y0:y_batch})

        if iteration % 100 == 0:
            mse = loss.eval(feed_dict{X0:X_batch,Y0:y_batch})
            print(iteration,"\tMSE:",mse)
