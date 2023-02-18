from __future__ import print_function
import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy.io as scio
import time
from datetime import datetime
from scipy.misc import imsave
import scipy.ndimage
import math
from skimage import img_as_ubyte
from utils import *

WEIGHT_INIT_STDDEV = 0.03
lambda_smooth = 0.01

eps = 1e-20
tau = 0.5

class Affine_Generator(object):
	def __init__(self, scope_name):
		self.scope = scope_name

	def field_model(self, img_a, img_b):
		with tf.variable_scope(self.scope):
			x = tf.concat([img_a, img_b], axis=-1)

			with tf.variable_scope('layer1'):
				weights = tf.get_variable("w1", [15, 15, 2, 16],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b1", [16], initializer=tf.constant_initializer(0.0))
				conv1 = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
				conv1 = lrelu(conv1)
				# conv1_r= resblock(conv1_a, 32)
				conv1_offset = convoffset2D(conv1, channel=16, scope_name='conv1_offset')
				bias = tf.get_variable("offset_b1", [16], initializer=tf.constant_initializer(0.0))
				conv1_offset = lrelu(conv1_offset + bias)
				# conv1_offset = convoffset2D(conv1_r, scope_name = 'conv3_offset')
				size1 = conv1_offset.shape

			with tf.variable_scope('layer2'):
				weights = tf.get_variable("w2", [15, 15, 16, 32],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b2", [32], initializer=tf.constant_initializer(0.0))
				conv2 = tf.nn.conv2d(conv1_offset, weights, strides=[1, 2, 2, 1], padding='SAME') + bias
				conv2 = lrelu(conv2)
				# conv2 = resblock(conv2, 64)
				conv2_offset = convoffset2D(conv2, channel=32, scope_name='conv2_offset')
				bias = tf.get_variable("offset_b2", [32], initializer=tf.constant_initializer(0.0))
				conv2_offset = lrelu(conv2_offset + bias)
				conv2_pool = tf.nn.max_pool(conv2_offset, ksize=[1, pool_size, pool_size, 1],
											strides=[1, pool_size, pool_size, 1], padding='SAME')

			with tf.variable_scope('layer3'):
				weights = tf.get_variable("w3", [15, 15, 32, 32],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b3", [32], initializer=tf.constant_initializer(0.0))
				conv3 = tf.nn.conv2d(conv2_pool, weights, strides=[1, 2, 2, 1], padding='SAME') + bias
				conv3 = lrelu(conv3)
				# conv3_r = resblock(conv3, 64)
				conv3_offset = convoffset2D(conv3, channel=32, scope_name='conv3_offset')
				bias = tf.get_variable("offset_b3", [32], initializer=tf.constant_initializer(0.0))
				conv3_offset = lrelu(conv3_offset + bias)
				conv3_pool = tf.nn.max_pool(conv3_offset, ksize=[1, pool_size, pool_size, 1],
											strides=[1, pool_size, pool_size, 1], padding='SAME')

			with tf.variable_scope('layer4'):
				weights = tf.get_variable("w4", [15, 15, 32, 32],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b4", [32], initializer=tf.constant_initializer(0.0))
				conv4 = tf.nn.conv2d(conv3_pool, weights, strides=[1, 2, 2, 1], padding='SAME') + bias
				conv4 = lrelu(conv4)
				conv4_offset = convoffset2D(conv4, channel=32, scope_name='conv4_offset')
				bias = tf.get_variable("offset_b4", [32], initializer=tf.constant_initializer(0.0))
				conv4_offset = lrelu(conv4_offset + bias)
				# conv4_pool = tf.nn.max_pool(conv4_r, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')
				conv4_pool = tf.nn.max_pool(conv4_offset, ksize=[1, pool_size, pool_size, 1],
											strides=[1, pool_size, pool_size, 1], padding='SAME')

			with tf.variable_scope('layer5'):
				weights = tf.get_variable("w5", [15, 15, 32, 64],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b5", [64], initializer=tf.constant_initializer(0.0))
				conv5 = tf.nn.conv2d(conv4_pool, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
				conv5 = lrelu(conv5)
				conv5_offset = convoffset2D(conv5, channel=64, scope_name='conv5_offset')
				bias = tf.get_variable("offset_b5", [64], initializer=tf.constant_initializer(0.0))
				conv5_offset = conv5_offset + bias # lrelu(conv5_offset + bias)
				# conv4_pool = tf.nn.max_pool(conv4_r, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')
				# conv4_r, attention_map = attention(x = conv4_r, ch = conv4_r.shape[-1].value, sn = False, scope_name = 'sa',
				#                                name = 'self_attention')
				size5 = conv5_offset.shape
				conv5_pool = tf.nn.max_pool(conv5_offset, ksize=[1, pool_size, pool_size, 1],
											strides=[1, pool_size, pool_size, 1], padding='SAME')

			with tf.variable_scope('fcn'):
				output = tf.layers.dense(inputs=gap, units=6, activation=None)

		return output

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv',
		 reuse=False):
	with tf.variable_scope(scope):
		if pad > 0:
			if (kernel - stride) % 2 == 0:
				pad_top = pad
				pad_bottom = pad
				pad_left = pad
				pad_right = pad

			else:
				pad_top = pad
				pad_bottom = kernel - stride - pad_top
				pad_left = pad
				pad_right = kernel - stride - pad_left

			if pad_type == 'zero':
				x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
			if pad_type == 'reflect':
				x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
		x = tf.layers.conv2d(inputs=x, filters=channels,
							 kernel_size=kernel, kernel_initializer=weight_init,
							 kernel_regularizer=weight_regularizer,
							 strides=stride, use_bias=use_bias, reuse=reuse)
		return x

def get_activation(activation):
	"""Get the appropriate activation from the given name"""
	if activation == 'relu':
		return nn.ReLU(inplace=False)
	elif activation == 'leaky_relu':
		negative_slope = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']
		return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
	elif activation == 'tanh':
		return nn.Tanh()
	elif activation == 'sigmoid':
		return nn.Sigmoid()
	else:
		return None

def convoffset2D(x, channel, scope_name=None):
	x_shape = tf.shape(x)
	x_shape_list = x.get_shape().as_list()
	with tf.variable_scope(scope_name):
		with tf.variable_scope('offset_conv'):
			offsets = conv(x, channels = channel * 2, kernel = 3, stride = 1, pad = 1, pad_type = 'reflect',
			               scope = 'conv', use_bias = False)
		with tf.variable_scope('weight_conv'):
			weights = conv(x, channels = channel, kernel = 3, stride = 1, pad = 1, pad_type = 'reflect',
			               scope = 'conv', use_bias = False)
			weights = tf.nn.sigmoid(weights)

		x = to_bc_h_w(x, x_shape)
		offsets = to_bc_h_w_2(offsets, x_shape)
		weights = to_bc_h_w(weights, x_shape)
		x_offset = deform_conv.tf_batch_map_offsets(x, offsets)
		weights = tf.expand_dims(weights, axis = 1)
		weights = to_b_h_w_c(weights, x_shape)
		x_offset = to_b_h_w_c(x_offset, x_shape)
		x_offset = tf.multiply(x_offset, weights)
		x_offset.set_shape(x_shape_list)
	return x_offset

def to_bc_h_w_2(x, x_shape):
	"""(b, h, w, 2c) -> (b*c, h, w, 2)"""
	x = tf.transpose(x, [0, 3, 1, 2])
	x = tf.reshape(x, [x_shape[0], x_shape[3], 2, x_shape[1], x_shape[2]])
	x = tf.transpose(x, [0, 1, 3, 4, 2])
	x = tf.reshape(x, [-1, x_shape[1], x_shape[2], 2])
	return x

def to_bc_h_w(x, x_shape):
	"""(b, h, w, c) -> (b*c, h, w)"""
	x = tf.transpose(x, [0, 3, 1, 2])
	x = tf.reshape(x, [-1, x_shape[1], x_shape[2]])
	return x

def to_b_h_w_c(x, x_shape):
	"""(b*c, h, w) -> (b, h, w, c)"""
	x = tf.reshape(x, (-1, x_shape[3], x_shape[1], x_shape[2]))
	x = tf.transpose(x, [0, 2, 3, 1])
	return x

