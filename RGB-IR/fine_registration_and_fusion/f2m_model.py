from __future__ import print_function
import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display

import scipy.io as scio
import time
from datetime import datetime
from scipy.misc import imsave
import scipy.ndimage
from skimage import img_as_ubyte
from utils import *
import sys
sys.path.append("..")

WEIGHT_INIT_STDDEV = 0.05
lambda_smooth = 0.01
pool_size = 2
eps = 1e-8

class F2M_Model(object):
	def __init__(self, BATCH_SIZE, INPUT_H, INPUT_W, is_training=False, BN=True):
		self.batchsize = BATCH_SIZE
		self.INPUT_H = INPUT_H
		self.INPUT_W = INPUT_W
		self.var_list_g = []
		self.step = 0
		self.lr = tf.placeholder(tf.float32, name='lr')
		self.is_training = is_training
		self.BN=BN

	def f2m(self, RGB, IR, defor_field=None, re_defor_field_gt=None):
		self.IR = IR
		self.RGB = RGB
		print(self.is_training)
		if self.is_training:
			with tf.device('/gpu:0'):
				self.defor_field =defor_field
				self.re_defor_field_gt = re_defor_field_gt
				x = tf.linspace(-1.0, 1.0, self.INPUT_W)
				y = tf.linspace(-1.0, 1.0, self.INPUT_H)
				xx, yy = tf.meshgrid(y, x)
				xx = tf.transpose(xx)
				yy = tf.transpose(yy)
				xx = tf.expand_dims(xx, -1)
				yy = tf.expand_dims(yy, -1)
				xx = tf.expand_dims(xx, 0)
				yy = tf.expand_dims(yy, 0)
				identity = tf.concat([yy, xx], axis=-1)
				identity = tf.tile(identity, [self.batchsize, 1, 1, 1])
				resampling_grid = identity + self.defor_field
				warped_R, _, _, _ = grid_sample(RGB[:, :, :, 0:1], resampling_grid)
				warped_G, _, _, _ = grid_sample(RGB[:, :, :, 1:2], resampling_grid)
				warped_B, _, _, _ = grid_sample(RGB[:, :, :, 2:3], resampling_grid)
				self.dRGB = tf.concat([warped_R, warped_G, warped_B], axis=-1)
		else:
			self.dRGB = RGB

		with tf.device('/gpu:0'):
			self.f2m_net = F2M_network('f2m_net', self.BN)
			drgb_ycbcr = rgb2ycbcr(self.dRGB)
			drgb_y = drgb_ycbcr[0: self.batchsize, :, :, 0:1]
			drgb_cb = drgb_ycbcr[0: self.batchsize, :, :, 1:2]
			drgb_cr = drgb_ycbcr[0: self.batchsize, :, :, 2:3]

			rgb_ycbcr = rgb2ycbcr(self.RGB)
			rgb_y = rgb_ycbcr[0: self.batchsize, :, :, 0:1]
			rgb_cb = rgb_ycbcr[0: self.batchsize, :, :, 1:2]
			rgb_cr = rgb_ycbcr[0: self.batchsize, :, :, 2:3]

			self.mean_img_before = ycbcr2rgb(tf.concat([(self.IR + rgb_y) / 2, rgb_cb, rgb_cr], axis=-1))
			self.mean_img_after = ycbcr2rgb(tf.concat([(self.IR + drgb_y) / 2, drgb_cb, drgb_cr], axis=-1))

			self.mean_y_before = (self.IR + rgb_y) / 2
			self.mean_y_after = (self.IR + drgb_y) / 2

			self.fimg_y, self.offset, self.a_offset = self.f2m_net.fuse(self.dRGB, self.IR)

			drgb_offset_ycbcr = rgb2ycbcr(self.a_offset)
			drgb_offset_y = drgb_offset_ycbcr[0: self.batchsize, :, :, 0:1]
			drgb_offset_cb = drgb_offset_ycbcr[0: self.batchsize, :, :, 1:2]
			drgb_offset_cr = drgb_offset_ycbcr[0: self.batchsize, :, :, 2:3]

			self.fused_img = ycbcr2rgb(tf.concat([self.fimg_y, drgb_offset_cb, drgb_offset_cr], axis=-1))

			if self.is_training:
				dr_defor_gt, _, _, _ = grid_sample(self.dRGB[0:self.batchsize, :, :, 0:1],
												   - (resampling_grid - identity) + identity)
				dg_defor_gt, _, _, _ = grid_sample(self.dRGB[0:self.batchsize, :, :, 1:2],
												   - (resampling_grid - identity) + identity)
				db_defor_gt, _, _, _ = grid_sample(self.dRGB[0:self.batchsize, :, :, 2:3],
												   - (resampling_grid - identity) + identity)
				self.drgb_defor_gt = tf.concat([dr_defor_gt, dg_defor_gt, db_defor_gt], axis=-1)

				self.sim_loss = 1 - 0.5 * SSIM(self.fimg_y, self.IR) - 0.5 * SSIM(self.fimg_y, drgb_offset_y)
				enhanced_gradient_drgb_offset_y = tf.where(
					gradient(drgb_offset_y) > tf.zeros_like(gradient(drgb_offset_y)),
					tf.pow(gradient(drgb_offset_y), 0.7), -tf.pow(-gradient(drgb_offset_y), 0.7))
				enhanced_gradient_IR = tf.where(gradient(self.IR) > tf.zeros_like(gradient(self.IR)),
												tf.pow(gradient(self.IR), 0.7), -tf.pow(-gradient(self.IR), 0.7))
				self.max_gradient_loss = l1(gradient(self.fimg_y),
											tf.where(tf.abs(gradient(drgb_offset_y)) > tf.abs(gradient(self.IR)),
													 enhanced_gradient_drgb_offset_y, enhanced_gradient_IR))
				self.fuse_loss = self.sim_loss + 10 * self.max_gradient_loss
				self.smooth_loss = smoothness_loss(self.offset, tf.ones_like(drgb_offset_y))
				self.defor_diff = l2(self.re_defor_field_gt, self.offset)
				self.fimg_grad_loss = l0(self.fimg_y)
				self.offset_loss = self.defor_diff + 0.001 * self.fimg_grad_loss


	def get_identity_grid(self):
		"""Returns a sampling-grid that represents the identity transformation."""
		x = tf.linspace(-1.0, 1.0, self.ow)  #
		y = tf.linspace(-1.0, 1.0, self.oh)  #
		xx, yy = tf.meshgrid([y, x])  #
		xx = tf.transpose(xx)
		yy = tf.transpose(yy)
		xx = xx.unsqueeze(dim=0)
		yy = yy.unsqueeze(dim=0)
		identity = tf.concat((yy, xx), dim=0).unsqueeze(0)  #
		return identity


WEIGHT_INIT_STDDEV = 0.03
lambda_smooth = 0.01
eps = 1e-20

class F2M_network(object):
	def __init__(self, scope_name, BN):
		self.scope = scope_name
		self.BN = BN

	def fuse(self, img_a, img_b, reuse=False):
		with tf.variable_scope(self.scope, reuse=reuse):
			a_offset, self.offset = convoffset2D(img_a, img_b, scope_name='conv1_offset', BN=self.BN)

			weights = tf.compat.v1.get_variable("w_a1", [1, 1, 3, 8],
												initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias = tf.compat.v1.get_variable("b_a1", [8], initializer=tf.constant_initializer(0.0))

			conv_a1 = tf.nn.conv2d(a_offset, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
			conv_a1 = lrelu(conv_a1)

			weights = tf.compat.v1.get_variable("w_b1", [1, 1, 1, 8],
												initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias = tf.compat.v1.get_variable("b_b1", [8], initializer=tf.constant_initializer(0.0))
			conv_b1 = tf.nn.conv2d(img_b, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
			conv_b1 = lrelu(conv_b1)

			weights = tf.compat.v1.get_variable("w_a2", [3, 3, 8, 16],
												initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias = tf.get_variable("b_a2", [16], initializer=tf.constant_initializer(0.0))
			conv_a2 = tf.nn.conv2d(conv_a1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
			conv_a2 = lrelu(conv_a2)

			weights = tf.compat.v1.get_variable("w_b2", [3, 3, 8, 16],
												initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias = tf.get_variable("b_b2", [16], initializer=tf.constant_initializer(0.0))
			conv_b2 = tf.nn.conv2d(conv_b1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
			conv_b2 = lrelu(conv_b2)

			# a_offset2, self.offset2 = convoffset2D(conv_a2, conv_b2, scope_name='conv2_offset')

			weights = tf.compat.v1.get_variable("w_a3", [3, 3, 16, 32],
												initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias = tf.compat.v1.get_variable("b_a3", [32], initializer=tf.constant_initializer(0.0))
			conv_a3 = tf.nn.conv2d(conv_a2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
			conv_a3 = lrelu(conv_a3)

			weights = tf.compat.v1.get_variable("w_b3", [3, 3, 16, 32],
												initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias = tf.compat.v1.get_variable("b_b3", [32], initializer=tf.constant_initializer(0.0))
			conv_b3 = tf.nn.conv2d(conv_b2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
			conv_b3 = lrelu(conv_b3)
			# conv_b3 = tf.concat([conv_b3, a_offset1_grad], axis=-1)
			# conv_a3, conv_b3, self.scale = channelattention(conv_a3, conv_b3, scope='ca')

			conv_a3, conv_b3 = channelattention(conv_a3, conv_b3, scope='ca')

			weights = tf.compat.v1.get_variable("wf1", [3, 3, 32, 32],
												initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias = tf.compat.v1.get_variable("bf1", [32], initializer=tf.constant_initializer(0.0))
			fused = tf.concat([conv_a3, conv_b3], axis=-1) # conv_a3 + conv_b3  # tf.where(conv_a3>conv_b3, conv_a3, conv_b3)
			x = tf.nn.conv2d(fused, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
			x = lrelu(x)

			weights = tf.compat.v1.get_variable("wf2", [3, 3, 32, 16],
												initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias = tf.compat.v1.get_variable("bf2", [16], initializer=tf.constant_initializer(0.0))
			x = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
			x = lrelu(x)

			weights = tf.compat.v1.get_variable("wf3", [3, 3, 16, 1],
												initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias = tf.compat.v1.get_variable("bf3", [1], initializer=tf.constant_initializer(0.0))
			x = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
			x = tf.nn.tanh(x) / 2 + 0.5

		return x, self.offset, a_offset



def img_resize(x, scale):
	oh = x.shape[1]
	ow = x.shape[2]
	x = tf.image.resize_images(images=x, size=[tf.constant(oh * scale), tf.constant(ow * scale)])
	return x

def avg_pool(x):
	return tf.nn.avg_pool(x, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

def res_block(x_init, channels, use_bias=True, sn=False, scope='resblock', reuse=False):
	with tf.variable_scope(scope):
		with tf.variable_scope('res_conv1'):
			x = conv(x_init, channels, kernel=5, stride=1, pad=2, pad_type='reflect', use_bias=use_bias, reuse=reuse)
			x = relu(x)

		with tf.variable_scope('res_conv2'):
			x = conv(x, channels, kernel=5, stride=1, pad=2, pad_type='reflect', use_bias=use_bias, reuse=reuse)

		return x + x_init


def us(x, ratio=2):
	_, height, width, _ = x.shape
	x_d = tf.image.resize_images(images=x, size=[tf.constant(int(height * ratio)), tf.constant(int(width * ratio))],
								 method=1)
	return x_d


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv', reuse=False):
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
		x = tf.layers.conv2d(inputs=x, filters=channels,
							 kernel_size=kernel,
							 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
							 kernel_regularizer=None,
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


def convoffset2D(x, y, scope_name=None, BN=True):
	x_shape = tf.shape(x)
	x_shape_list = x.get_shape().as_list()
	channel = x_shape_list[-1]
	batchsize = x_shape_list[0]
	oh = x_shape_list[1]
	ow = x_shape_list[2]
	dx = tf.linspace(-1.0, 1.0, ow)
	dy = tf.linspace(-1.0, 1.0, oh)
	xx, yy = tf.meshgrid(dy, dx)  #
	xx = tf.transpose(xx)
	yy = tf.transpose(yy)
	xx = tf.expand_dims(xx, -1)
	yy = tf.expand_dims(yy, -1)
	xx = tf.expand_dims(xx, 0)
	yy = tf.expand_dims(yy, 0)
	identity = tf.concat([yy, xx], axis=-1)
	identity = tf.tile(identity, [batchsize, 1, 1, 1])

	with tf.variable_scope(scope_name):
		with tf.variable_scope('offset_conv'):
			offsets1 = conv(tf.concat([x, y], axis=-1), channels=32, kernel=3, stride=1, pad=1, pad_type='reflect',
							scope='conv1', use_bias=True)
			offsets1 = tf.contrib.layers.batch_norm(inputs=offsets1, activation_fn=None, is_training=BN)
			offsets1 = lrelu(offsets1)
			# offsets1 = res_block(offsets1, channels=32, scope='res1')
			offsets1_ds = tf.nn.max_pool(offsets1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

			offsets2 = conv(offsets1_ds, channels=32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv2', use_bias=True)
			#offsets2 = tf.contrib.layers.batch_norm(inputs=offsets2, activation_fn=None, is_training=is_training)
			offsets2 = lrelu(offsets2)
			# offsets2 = res_block(offsets2, channels=32, scope='res2')
			offsets2_ds = tf.nn.max_pool(offsets2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')


			offsets3 = conv(offsets2_ds, channels=32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv3', use_bias=True)
			offsets3 = tf.contrib.layers.batch_norm(inputs=offsets3, activation_fn=None, is_training=BN)
			offsets3 = lrelu(offsets3)
			# offsets3 = res_block(offsets3, channels=32, scope='res3')
			offsets3_ds = tf.nn.max_pool(offsets3, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')


			offsets5 = conv(offsets3_ds, channels=32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv5',
							use_bias=True)
			offsets5 = tf.contrib.layers.batch_norm(inputs=offsets5, activation_fn=None, is_training=BN)
			offsets5 = lrelu(offsets5)
			# offsets5 = res_block(offsets5, channels=32, scope='res5')


			offsets6 = conv(tf.concat([us(offsets5), offsets3], axis=-1), channels=32, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv6', use_bias=True)
			offsets6 = tf.contrib.layers.batch_norm(inputs=offsets6, activation_fn=None, is_training=BN)
			offsets6 = lrelu(offsets6)

			offsets7 = conv(tf.concat([us(offsets6), offsets2], axis=-1), channels=32, kernel=3, stride=1, pad=1,
							pad_type='reflect', scope='conv7', use_bias=True)
			offsets7 = tf.contrib.layers.batch_norm(inputs=offsets7, activation_fn=None, is_training=BN)
			offsets7 = lrelu(offsets7)

			offsets8 = conv(tf.concat([us(offsets7), offsets1], axis=-1), channels=32, kernel=3, stride=1, pad=1,
							pad_type='reflect', scope='conv8', use_bias=True)
			offsets8 = tf.contrib.layers.batch_norm(inputs=offsets8, activation_fn=None, is_training=BN)
			offsets8 = lrelu(offsets8)

			offsets10 = conv(offsets8, channels=16, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv10', use_bias=True)
			offsets10 = tf.contrib.layers.batch_norm(inputs=offsets10, activation_fn=None, is_training=BN)
			offsets10 = lrelu(offsets10)

			offsets = conv(offsets10, channels=2, kernel=3, stride=1, pad=1, pad_type='reflect',
						   scope='conv11', use_bias=True)
			offsets = tf.nn.tanh(offsets) * 0.2
		x_offset = grid_sample_channels(x, offsets + identity)  # offsets + identity

	return x_offset, offsets


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


def tf_flatten(a):
	"""Flatten tensor"""
	return tf.reshape(a, [-1])


def tf_repeat(a, repeats, axis=0):
	a = tf.expand_dims(a, -1)
	a = tf.tile(a, [1, repeats])
	a = tf_flatten(a)
	return a


def tf_map_coordinates(input, coords):
	"""
	:param input: tf.Tensor. shape=(h, w)
	:param coords: tf.Tensor. shape = (n_points, 2)
	:return:
	"""
	coords_tl = tf.cast(tf.floor(coords), tf.int32)
	coords_br = tf.cast(tf.ceil(coords), tf.int32)
	coords_bl = tf.stack([coords_br[:, 0], coords_tl[:, 1]], axis=1)
	coords_tr = tf.stack([coords_tl[:, 0], coords_br[:, 1]], axis=1)
	vals_tl = tf.gather_nd(input, coords_tl)
	vals_br = tf.gather_nd(input, coords_br)
	vals_bl = tf.gather_nd(input, coords_bl)
	vals_tr = tf.gather_nd(input, coords_tr)
	coords_offset_tl = coords - tf.cast(coords_tl, tf.float32)
	vals_t = vals_tl + (vals_tr - vals_tl) * coords_offset_tl[:, 1]
	vals_b = vals_bl + (vals_br - vals_bl) * coords_offset_tl[:, 1]
	mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_tl[:, 0]
	return mapped_vals


def tf_batch_map_coordinates(input, coords):
	"""
	Batch version of tf_map_coordinates
	:param input: tf.Tensor. shape = (b, h, w)
	:param coords: tf.Tensor. shape = (b, n_points, 2)
	:return:
	"""
	input_shape = tf.shape(input)
	batch_size = input_shape[0]
	input_size_h = input_shape[1]
	input_size_w = input_shape[2]
	n_coords = tf.shape(coords)[1]
	coords_w = tf.clip_by_value(coords[..., 1], 0, tf.cast(input_size_w, tf.float32) - 1)
	coords_h = tf.clip_by_value(coords[..., 0], 0, tf.cast(input_size_h, tf.float32) - 1)
	coords = tf.stack([coords_h, coords_w], axis=-1)
	coords_tl = tf.cast(tf.floor(coords), tf.int32)
	coords_br = tf.cast(tf.ceil(coords), tf.int32)
	coords_bl = tf.stack([coords_br[..., 0], coords_tl[..., 1]], axis=-1)
	coords_tr = tf.stack([coords_tl[..., 0], coords_br[..., 1]], axis=-1)
	idx = tf_repeat(tf.range(batch_size), n_coords)

	def _get_vals_by_coords(input, coords):
		indices = tf.stack([idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])], axis=-1)
		vals = tf.gather_nd(input, indices)
		vals = tf.reshape(vals, (batch_size, n_coords))
		return vals

	vals_tl = _get_vals_by_coords(input, coords_tl)
	vals_br = _get_vals_by_coords(input, coords_br)
	vals_bl = _get_vals_by_coords(input, coords_bl)
	vals_tr = _get_vals_by_coords(input, coords_tr)

	coords_offset_tl = coords - tf.cast(coords_tl, 'float32')
	vals_t = vals_tl + (vals_tr - vals_tl) * coords_offset_tl[..., 1]
	vals_b = vals_bl + (vals_br - vals_bl) * coords_offset_tl[..., 1]
	mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_tl[..., 0]

	return mapped_vals


def tf_batch_map_offsets(input, offsets):
	"""
	:param input: tf.Tensor, shape=(b, h, w)
	:param offsets: tf.Tensor, shape=(b, h, w, 2)
	:return:
	"""
	input_shape = tf.shape(input)
	batch_size = input_shape[0]
	input_size_h = input_shape[1]
	input_size_w = input_shape[2]
	offsets = tf.reshape(offsets, (batch_size, -1, 2))
	grid_x, grid_y = tf.meshgrid(tf.range(input_size_w), tf.range(input_size_h))
	grid = tf.stack([grid_y, grid_x], axis=-1)
	grid = tf.cast(grid, tf.float32)
	grid = tf.reshape(grid, (-1, 2))
	grid = tf.expand_dims(grid, axis=0)
	grid = tf.tile(grid, multiples=[batch_size, 1, 1])
	coords = offsets + grid
	mapped_vals = tf_batch_map_coordinates(input, coords)
	return mapped_vals


def tf_batch_map_offsets_channels(input, offsets):
	"""
	:param input: tf.Tensor, shape=(b, h, w, C)
	:param offsets: tf.Tensor, shape=(b, h, w, 2)
	:return:
	"""
	input_shape = input.shape
	channels = input_shape[-1]
	batchsize = input_shape[0]

	for c in range(channels):
		x = input[0: batchsize, :, :, c:c + 1]
		x_shape = tf.shape(x)
		x = to_bc_h_w(x, x_shape)
		offsets = to_bc_h_w_2(offsets, x_shape)

		offset_channel = tf_batch_map_offsets(x, offsets)
		if c == 0:
			result = offset_channel
		else:
			result = tf.concat([result, offset_channel], axis=-1)
	return result


def NCC(img1, img2):
	h = img1.shape[1]
	w = img2.shape[2]
	mean1 = tf.reduce_mean(img1, axis=[1, 2])
	mean2 = tf.reduce_mean(img2, axis=[1, 2])
	mean1 = tf.expand_dims(mean1, axis=1)
	mean1 = tf.expand_dims(mean1, axis=2)
	mean2 = tf.expand_dims(mean2, axis=1)
	mean2 = tf.expand_dims(mean2, axis=2)
	mean1 = tf.tile(mean1, [1, int(h), int(w), 1])
	mean2 = tf.tile(mean2, [1, int(h), int(w), 1])
	dimg1 = img1 - mean1
	dimg2 = img2 - mean2
	numerator = tf.multiply(dimg1, dimg2)

	std1 = reduce_std(img1, axis=[1, 2], keepdims=True)
	std1 = tf.tile(std1, [1, int(h), int(w), 1])
	std2 = reduce_std(img2, axis=[1, 2], keepdims=True)
	std2 = tf.tile(std2, [1, int(h), int(w), 1])
	denominator = tf.multiply(std1, std2)

	return tf.reduce_mean(tf.divide(numerator, denominator))


def reduce_std(x, axis=None, keepdims=False):
	return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
	m = tf.reduce_mean(x, axis=axis, keep_dims=True)
	devs_squared = tf.square(x - m)
	return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def normalize_common_tf(a, b):
	B, h, w, c = a.shape
	a_min = tf.expand_dims(tf.reduce_min(a, axis=[1, 2, 3]), axis=-1)
	a_max = tf.expand_dims(tf.reduce_max(a, axis=[1, 2, 3]), axis=-1)
	b_min = tf.expand_dims(tf.reduce_min(b, axis=[1, 2, 3]), axis=-1)
	b_max = tf.expand_dims(tf.reduce_max(b, axis=[1, 2, 3]), axis=-1)
	minimum = tf.reduce_min(tf.concat([a_min, b_min], axis=-1), axis=-1)
	maximum = tf.reduce_max(tf.concat([a_max, b_max], axis=-1), axis=-1)
	minimum = tf.expand_dims(tf.expand_dims(tf.expand_dims(minimum, axis=-1), axis=-1), axis=-1)
	maximum = tf.expand_dims(tf.expand_dims(tf.expand_dims(maximum, axis=-1), axis=-1), axis=-1)
	minimum = tf.tile(minimum, [1, h, w, c])
	maximum = tf.tile(maximum, [1, h, w, c])
	new_a = tf.clip_by_value((a - minimum) / (maximum - minimum), 1e-10, 1.0)
	new_b = tf.clip_by_value((b - minimum) / (maximum - minimum), 1e-10, 1.0)
	return new_a, new_b


def normalize(a):
	b, h, w, c = a.shape
	a_min = tf.reshape(tf.reduce_min(a, axis=[1, 2, 3]), [b, 1, 1, 1])
	a_max = tf.reshape(tf.reduce_max(a, axis=[1, 2, 3]), [b, 1, 1, 1])
	a_min = tf.tile(a_min, [1, h, w, c])
	a_max = tf.tile(a_max, [1, h, w, c])
	new_a = tf.clip_by_value((a - a_min) / (a_max - a_min), 1e-10, 1.0)
	return new_a


def grid_sample_channels(input, grid):
	input_shape = input.shape
	channels = input_shape[-1]
	batchsize = input_shape[0]

	for c in range(channels):
		x = input[0:batchsize, :, :, c:c + 1]
		offset_channel, _, _, _ = grid_sample(x, grid)
		if c == 0:
			result = offset_channel
		else:
			result = tf.concat([result, offset_channel], axis=-1)
	return result
