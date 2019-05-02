import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
import skimage
import scipy
import os
import glob
import sys
import random

bbox_tensor_size = 4 # x,y,width,height
k = 2 # detect radius

conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

pool5_map = 7

laplace_lim_width = 0.6
laplace_lim_height = 1.4



def layers(prevX,currX):
    with tf.variable_scope("prev_layer"):
        prev_conv1 = tf.layers.conv2d(prevX, filters=64, kernel_size=3,
                                 strides=1, padding=conv1_pad,
                                 activation=tf.nn.relu, name="conv1")
        prev_conv2 = tf.layers.conv2d(prev_conv1, filters=64, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv2")
        prev_pool1 = tf.nn.max_pool(prev_conv2, ksize=[1, 2, 2, 1], strides=[1,2,2, 1], padding="SAME")

        
        prev_conv3 = tf.layers.conv2d(prev_pool1, filters=128, kernel_size=3,
                                 strides=1, padding=conv1_pad,
                                 activation=tf.nn.relu, name="conv3")
        prev_conv4 = tf.layers.conv2d(prev_conv3, filters=128, kernel_size=conv2_ksize,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv4")
        prev_pool2 = tf.nn.max_pool(prev_conv4, ksize=[1, 2, 2, 1], strides=[1,2,2, 1], padding="SAME")

        
        prev_conv5 = tf.layers.conv2d(prev_pool2, filters=256, kernel_size=3,
                                 activation=tf.nn.relu, name="conv5")
        prev_conv6 = tf.layers.conv2d(prev_conv5, filters=256, kernel_size=3,
                                 activation=tf.nn.relu, name="conv6")
        prev_conv7 = tf.layers.conv2d(prev_conv6, filters=256, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv7")
        prev_pool3 = tf.nn.max_pool(prev_conv7, ksize=[1,2, 2, 1], strides=[1, 2,2, 1], padding="SAME")


        prev_conv8 = tf.layers.conv2d(prev_pool3, filters=512, kernel_size=3,
                                 strides=1, padding=conv1_pad,
                                 activation=tf.nn.relu, name="conv8")
        prev_conv9 = tf.layers.conv2d(prev_conv8, filters=512, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv9")
        prev_conv10 = tf.layers.conv2d(prev_conv9, filters=512, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv10")
        prev_pool4 = tf.nn.max_pool(prev_conv10, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding="SAME")


        prev_conv11 = tf.layers.conv2d(prev_pool4, filters=512, kernel_size=3,
                                 strides=1, padding=conv1_pad,
                                 activation=tf.nn.relu, name="conv11")
        prev_conv12 = tf.layers.conv2d(prev_conv11, filters=512, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv12")
        prev_conv13 = tf.layers.conv2d(prev_conv12, filters=512, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv13")
        prev_pool5 = tf.nn.max_pool(prev_conv13, ksize=[1, 2, 2, 1], strides=[1,2,2, 1], padding="SAME")

    with tf.variable_scope("current_layer"):
        conv1 = tf.layers.conv2d(currX, filters=64, kernel_size=3,
                                 strides=1, padding=conv1_pad,
                                 activation=tf.nn.relu, name="conv1")
        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv2")
        pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1,2,2, 1], padding="SAME")

        
        conv3 = tf.layers.conv2d(pool1, filters=128, kernel_size=3,
                                 strides=1, padding=conv1_pad,
                                 activation=tf.nn.relu, name="conv3")
        conv4 = tf.layers.conv2d(conv3, filters=128, kernel_size=conv2_ksize,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv4")
        pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1,2,2, 1], padding="SAME")

        
        conv5 = tf.layers.conv2d(pool2, filters=256, kernel_size=3,
                                 activation=tf.nn.relu, name="conv5")
        conv6 = tf.layers.conv2d(conv5, filters=256, kernel_size=3,
                                 activation=tf.nn.relu, name="conv6")
        conv7 = tf.layers.conv2d(conv6, filters=256, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv7")
        pool3 = tf.nn.max_pool(conv7, ksize=[1,2, 2, 1], strides=[1, 2,2, 1], padding="SAME")


        conv8 = tf.layers.conv2d(pool3, filters=512, kernel_size=3,
                                 strides=1, padding=conv1_pad,
                                 activation=tf.nn.relu, name="conv8")
        conv9 = tf.layers.conv2d(conv8, filters=512, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv9")
        conv10 = tf.layers.conv2d(conv9, filters=512, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv10")
        pool4 = tf.nn.max_pool(conv10, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding="SAME")


        conv11 = tf.layers.conv2d(pool4, filters=512, kernel_size=3,
                                 strides=1, padding=conv1_pad,
                                 activation=tf.nn.relu, name="conv11")
        conv12 = tf.layers.conv2d(conv11, filters=512, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv12")
        conv13 = tf.layers.conv2d(conv12, filters=512, kernel_size=3,
                                 strides=1, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv13")
        pool5 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1,2,2, 1], padding="SAME")
        concat =  tf.concat([pool5,prev_pool5],1)

    return concat# return concated feature


def prediction(feature,training):
    with tf.name_scope("vgg_dnn"):
        feature_flat = tf.reshape(feature, shape=[-1, 14 * 7 * 512])
        fcn = tf.layers.dense(feature_flat,1024, activation=tf.nn.relu, name='fcn1', kernel_initializer= tf.random_normal_initializer(stddev=0.01))
        fcn = tf.layers.dropout(fcn, tf.Variable(0.05), training=training)
        fcn = tf.layers.dense(fcn,1024,activation=tf.nn.relu, name='fcn2', kernel_initializer= tf.random_normal_initializer(stddev=0.01))
        fcn = tf.layers.dropout(fcn, tf.Variable(0.05), training=training)
        pred_bbox = tf.layers.dense(fcn,bbox_tensor_size, name='pred')
    return pred_bbox

def load_weights(sess,saver,filepath):
    saver.restore(sess,filepath)
