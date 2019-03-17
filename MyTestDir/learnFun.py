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
tensor1=[[1,5,5],[3,2,2]]
tensor2=tf.convert_to_tensor(tensor1)
print tf.Session().run(tensor2)
