import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
from utils import *

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
weight_regularizer = None

class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name

	def encode(self, img, reuse=False):
		with tf.compat.v1.variable_scope(self.scope, reuse=reuse):
			input = img

			x1 = conv(input, 32, kernel=1, stride=1, pad=0, pad_type='reflect', scope='conv1')
			x1 = instance_norm(x1, scope='in1') #x = tf.layers.batch_normalization(x, training=True)
			x1 = lrelu(x1)

			x2 = conv(x1, 32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv2')
			x2 = instance_norm(x2, scope='in2')
			x2 = lrelu(x2)
			x2 = con(x1, x2)

			x2_us = tf.nn.max_pool(x2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

			# x2 = resblock(x1, 32, use_bias=True, sn=False, scope='resblock2', reuse=False)

			x3 = conv(x2_us, 64, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv3')
			x3 = instance_norm(x3, scope='in3')
			x3 = lrelu(x3)

			x4 = conv(x3, 64, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv4')
			x4 = instance_norm(x4, scope='in4')
			x4 = lrelu(x4)
			x4 = con(x2_us, x4)

			x4_us = tf.nn.max_pool(x4, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

			x5 = up_layer('uplayer5', x4_us, 64, kernel_size=3)
			x5 = conv(con(x5, x4), 64, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv5')
			x5 = instance_norm(x5, scope='in5')
			x5 = lrelu(x5)

			x6 = up_layer('uplayer6', x5, 64, kernel_size=3)
			x6 = conv(con(x6, x2), 64, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv6')
			x6 = instance_norm(x6, scope='in6')
			x6 = lrelu(x6)

			x7 = conv(x6, 32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv7')
			x7 = instance_norm(x7, scope='in7')
			x7 = lrelu(x7)

			x = conv(x7, 1, kernel=1, stride=1, pad=0, pad_type='reflect', scope='conv8')
			# x = instance_norm(x, scope='in6')
			# x = tf.nn.tanh(x)/2 + 0.5
		return x, x7


class Shared_Layers(object):
	def __init__(self, scope_name):
		self.scope = scope_name

	def encode(self, input, code_dim, reuse=False):
		with tf.variable_scope(self.scope, reuse = reuse):
			x = conv(input, 64, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv1')
			x = lrelu(x)
			# 128

			x = conv(x, 128, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv2')
			x = lrelu(x)
			# 64

			x = conv(x, 64, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv3')
			x = lrelu(x)
			# 32

			x = conv(x, 64, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv4')
			x = lrelu(x)
			# 16

			x = tf.reshape(x, [-1, int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])])

			with tf.variable_scope('fc1'):
				x = tf.layers.dense(x, units=code_dim*2, kernel_initializer=weight_init,
									kernel_regularizer=weight_regularizer, use_bias=True, reuse=reuse)
				x = lrelu(x)
			with tf.variable_scope('fc2'):
				x = tf.layers.dense(x, units=code_dim, kernel_initializer=weight_init,
									kernel_regularizer=weight_regularizer, use_bias=True, reuse=reuse)
		return x

def con(x,y):
	return tf.concat([x,y],axis=-1)

def resblock(x_init, channels, use_bias=True, sn=False, scope='resblock', reuse=False):
	with tf.variable_scope(scope):
		with tf.variable_scope('res1'):
			x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn,
					 reuse=reuse)
			# x = instance_norm(x)
			x = lrelu(x)

		with tf.variable_scope('res2'):
			x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, reuse=reuse)
			# x = instance_norm(x)

		return x + x_init




def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv', reuse=False):
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

		if sn:
			w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
								regularizer=weight_regularizer)
			x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID',
							 reuse=reuse)
			if use_bias:
				bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
				x = tf.nn.bias_add(x, bias)

		else:
			x = tf.layers.conv2d(inputs=x, filters=channels,
								 kernel_size=kernel, kernel_initializer=weight_init,
								 kernel_regularizer=weight_regularizer,
								 strides=stride, use_bias=use_bias, reuse=reuse)
		return x


def instance_norm(x, scope='instance_norm'):
	# return tf.contrib.layers.instance_norm(x,
	#                                        epsilon=1e-05,
	#                                        center=True, scale=True,
	#                                        scope=scope)
	shape = x.get_shape().as_list()
	depth = shape[3]
	with tf.variable_scope(scope):
		scale = tf.compat.v1.get_variable("scale", shape=[depth],
								initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02, dtype=tf.float32))
		offset = tf.compat.v1.get_variable("offset", shape=[depth], initializer=tf.constant_initializer(0.0))
		mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
		epsilon = 1e-5
		inv = tf.math.rsqrt(variance + epsilon)
		normalized = (x - mean) * inv
	return scale * normalized + offset
