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

class Des_Extract_Model(object):
	def __init__(self, BATCH_SIZE, INPUT_H, INPUT_W, is_training=True, equivariance=True):
		self.batchsize = BATCH_SIZE
		self.var_list_g = []
		self.step = 0
		self.lr = tf.compat.v1.placeholder(tf.float32, name='lr')
		self.equivariance = equivariance
		self.is_training = is_training

	def des(self, RGB, IR):
		self.RGB = RGB
		self.IR = IR
		oh = IR.shape[1]
		ow = IR.shape[2]

		if self.equivariance:
			random_rotA = np.random.randint(4, size=self.batchsize)
			random_rotB = np.random.randint(4, size=self.batchsize)
			self.data_RGB = batch_rotate_p4(RGB, random_rotA)
			self.data_IR = batch_rotate_p4(IR, random_rotB)
		else:
			self.data_RGB = RGB
			self.data_IR = IR

		with tf.compat.v1.variable_scope('des_extract'):
			self.E_RGB = Encoder('RGB_Encoder')
			self.RGB_des, self.F_RGB = self.E_RGB.encode(self.data_RGB, reuse=False)
			self.E_IR = Encoder('IR_Encoder')
			self.IR_des, self.F_IR = self.E_IR.encode(self.data_IR, reuse=False)
		if self.equivariance:
			self.RGB_des = batch_rotate_p4(self.RGB_des, -random_rotA)
			self.IR_des = batch_rotate_p4(self.IR_des, -random_rotB)

		if self.is_training:
			RGB_ycbcr = rgb2ycbcr(self.RGB)
			RGB_y = RGB_ycbcr[:, :, :, 0:1]
			all_samples = tf.concat([self.RGB_des, self.IR_des], axis=0)

			N = self.batchsize
			for i in range(2 * N):
				s = similarity_eva(tf.tile(all_samples[i:i+1, :, :, 0:1], [2 * N, 1, 1, 1]), all_samples) / tau
				s = tf.expand_dims(s, axis=-1)
				if i==0:
					self.similarities = s
				else:
					self.similarities = tf.concat([self.similarities, s],axis=-1)

			# loss
			for i in range(2 * N):
				j = (i + N) % (2 * N)
				pos = self.similarities[i, j]
				neg_num=0
				for k in range(2*N):
					if k!=j:
						neg = tf.expand_dims(self.similarities[i, k], axis=0)
						if neg_num==0:
							neg_num = neg_num + 1
							negs = neg
						else:
							negs=tf.concat([negs, neg], axis=0)
				softmax = tf.expand_dims(-tf.reduce_logsumexp(pos) + tf.reduce_logsumexp(negs, axis=0), axis=-1)
				if i==0:
					softmaxes = softmax
				else:
					softmaxes = tf.concat([softmaxes, softmax], axis=0)

			self.contrast_loss = tf.reduce_mean(softmaxes)
			self.decay_loss = activation_decay([self.RGB_des, self.IR_des], p=2.) \
							  + activation_decay([self.RGB_des, self.IR_des], p=1.)
			self.loss = self.contrast_loss + 0.0001 * self.decay_loss




weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
weight_regularizer = None

class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name

	def encode(self, img, reuse=False):
		with tf.compat.v1.variable_scope(self.scope, reuse=reuse):
			input = img

			x1 = conv(input, 32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv1')
			x1 = instance_norm(x1, scope='in1') #x = tf.layers.batch_normalization(x, training=True)
			x1 = lrelu(x1)

			x2 = conv(x1, 32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv2')
			x2 = instance_norm(x2, scope='in2')
			x2 = lrelu(x2)
			x2 = tf.nn.max_pool(x2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

			x3 = conv(x2, 64, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv3')
			x3 = instance_norm(x3, scope='in3')
			x3 = lrelu(x3)

			x4 = conv(x3, 64, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv4')
			x4 = instance_norm(x4, scope='in4')
			x4 = lrelu(x4)
			x4 = tf.nn.max_pool(x4, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

			x5 = up_layer('uplayer5', x4, 64, kernel_size=3)
			x5 = conv(con(x5, x2), 64, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv5')
			x5 = instance_norm(x5, scope='in5')
			x5 = lrelu(x5)

			x6 = up_layer('uplayer6', x5, 64, kernel_size=3)
			x6 = conv(con(x6, x1), 64, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv6')
			x6 = instance_norm(x6, scope='in6')
			x6 = lrelu(x6)

			x7 = conv(x6, 32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv7')
			x7 = instance_norm(x7, scope='in7')
			x7 = lrelu(x7)

			x = conv(x7, 1, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv8')
		return x, x7

def con(x,y):
	return tf.concat([x,y],axis=-1)

def resblock(x_init, channels, use_bias=True, sn=False, scope='resblock', reuse=False):
	with tf.compat.v1.variable_scope(scope):
		with tf.variable_scope('res1'):
			x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn,
					 reuse=reuse)
			x = lrelu(x)

		with tf.variable_scope('res2'):
			x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, reuse=reuse)
		return x + x_init


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv', reuse=False):
	with tf.compat.v1.variable_scope(scope):
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
	with tf.compat.v1.variable_scope(scope):
		scale = tf.compat.v1.get_variable("scale", shape=[depth],
								initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02, dtype=tf.float32))
		offset = tf.compat.v1.get_variable("offset", shape=[depth], initializer=tf.constant_initializer(0.0))
		mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
		epsilon = 1e-5
		inv = tf.math.rsqrt(variance + epsilon)
		normalized = (x - mean) * inv
	return scale * normalized + offset


def similarity_eva(x, y):
	return tf.reduce_mean(-tf.square(x-y), axis=[1, 2, 3])

def activation_decay(tensors, p=2.):
	"""Computes the L_p^p norm over an activation map.
	"""
	loss = tf.constant(1.0, shape=[])
	Z = 0
	for tensor in tensors:
		Z += tf.size(tensor)
		loss += tf.reduce_sum(tf.abs(tf.pow(tensor, p)))
	return loss / tf.cast(Z, dtype=tf.float32)


def rgb2ycbcr(img_rgb):
	R = tf.expand_dims(img_rgb[:, :, :, 0], axis=-1)
	G = tf.expand_dims(img_rgb[:, :, :, 1], axis=-1)
	B = tf.expand_dims(img_rgb[:, :, :, 2], axis=-1)
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255
	img_ycbcr = tf.concat([Y, Cb, Cr], axis=-1)
	return img_ycbcr