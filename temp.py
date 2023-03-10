import tensorflow._api.v2.compat.v1 as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()

