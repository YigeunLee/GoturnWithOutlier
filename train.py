import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
import skimage
import scipy
import os
import glob

check_interval = 100
bbox_tensor_size = 4
 
def optimize(last_layer,correct_label,learning_rate):
    loss = tf.reduce_mean(tf.abs(tf.subtract(last_layer, correct_label)))
    optimizer = tf.train.AdamOptimizer(tf.Variable(learning_rate))
    training_op = optimizer.minimize(loss)
    return training_op,loss,0

def train(sess,prev_frame,curr_frame,label_bbox,training,training_op,loss,acc,n_epochs,get_batch_func,batch_size,train_set_filepath,training_val):
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs):
        prev_x,curr_x,bbox_label = get_batch_func(train_set_filepath)
        bbox_train_result = sess.run(training_op, feed_dict={prev_frame: prev_x,curr_frame:curr_x,label_bbox:bbox_label,training:training_val})
        
        if epoch % check_interval == 0:
            loss_val = loss.eval(feed_dict={prev_frame: prev_x,curr_frame:curr_x,label_bbox: bbox_label})
            #acc_val = acc.eval(feed_dict={prev_frame: prev_x,curr_frame:curr_x,label_bbox: bbox_label})
            print('loss:' + str(loss_val) + ' ' + 'epoch:' + str(epoch))

