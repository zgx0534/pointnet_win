#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print sess.run(output, feed_dict={input1: [3.], input2: [4.]})
