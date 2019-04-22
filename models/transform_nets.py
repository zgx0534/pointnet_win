# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    #expand_dims表示给数据扩展一个维度，-1表示在最后扩展一维
    #point_cloud本来是（32,1024,3）现在是（32,1024,3,1）
    input_image = tf.expand_dims(point_cloud, -1)
    # input_image （32,1024,3,1）
    # padding = 'VALID'表示不使用0补边，SAME表示用0补边
    # is_training 指明了是否为训练，训练时需要优化参数，测试时不需要
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    # net （32,1024,1,64）
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    # net （32,1024,1,128）
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    # net （32,1024,1,1024）

    #池化
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')
    # net （32,1,1,1024）
    net = tf.reshape(net, [batch_size, -1])
    # net （32,1024）
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    # net （32,512）
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)
    # net （32,256）
    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        # weights是一个[256,9]的全0矩阵
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        # biases是一个形状为[9，1]的全0矩阵
        biases  = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        # biases：[1,0,0,0,1,0,0,0,1]
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        # transform.shape (32, 9)
        transform = tf.nn.bias_add(transform, biases)
        # transform.shape (32, 9)
        # transform 是32行，每行都是[1,0,0,0,1,0,0,0,1]
    transform = tf.reshape(transform,[batch_size, 3, K])
    # transform.shape (32, 3, 3)
    # transform 变成32个[3，3]的单位矩阵
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    # (32,1024,1,64)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    # (32,1024,1,128)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    # (32,1024,1,1024)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')
    # (32,1,1,1024)
    net = tf.reshape(net, [batch_size, -1])
    # (32,1024)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)
    # (32, 256)
    with tf.variable_scope('transform_feat',reuse=True) as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        # weights :(256,4096) 全0
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        # biases :(4096,) 全0
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        # biases : 64*64的单位阵的铺平
        transform = tf.matmul(net, weights)
        # transform:(32,4096)
        transform = tf.nn.bias_add(transform, biases)
        # transform:(32,4096)

    transform = tf.reshape(transform, [batch_size, K, K])
    # transform 32个64*64的单位阵
    return transform
