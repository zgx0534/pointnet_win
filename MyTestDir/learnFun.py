# -*- coding: UTF-8 -*-
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
# k=3
# weights = tf.get_variable('weights', [32,256],initializer=tf.constant_initializer(2.0),dtype=tf.float32)
# #bias = tf.reshape(tf.constant([1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1], dtype=tf.float32),[2,3,3])
# aaa={}
# b=tf.constant(np.eye(64).flatten(), dtype=tf.float32)

transform = aaa['transform']
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(transform)