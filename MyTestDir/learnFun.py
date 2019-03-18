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
batch = tf.Variable(0)
tensor1=[[1,5,5],[3,2,2]]
tensor2=tf.convert_to_tensor(tensor1)
end_points = {1,}
end_points.add(5)

print end_points
