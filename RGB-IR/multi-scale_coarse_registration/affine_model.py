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
from task1_shared_information_extraction.des_extract_model import Des_Extract_Model

WEIGHT_INIT_STDDEV = 0.05
lambda_smooth = 0.01
pool_size = 2

eps = 1e-8

class Affine_Model(object):
	def __init__(self, BATCH_SIZE, INPUT_H, INPUT_W, is_training=False):
		self.batchsize = BATCH_SIZE
		self.INPUT_H = INPUT_H
		self.INPUT_W = INPUT_W
		self.var_list_g = []
		self.step = 0
		self.lr = tf.placeholder(tf.float32, name='lr')
		self.is_training = is_training

	def affine(self, RGB, IR):
		self.RGB = RGB
		self.IR = IR
		with tf.device('/gpu:0'):
			self.RGB_d4 = img_resize(self.RGB, 0.25)
			self.IR_d4 = img_resize(self.IR, 0.25)
			self.RGB_d2 = img_resize(self.RGB, 0.5)
			self.IR_d2 = img_resize(self.IR, 0.5)

			self.affine_net_d4 = Affine_Generator('affine_net_d4', self.is_training)
			dtheta_d4 = self.affine_net_d4.field_model(self.RGB_d4, self.IR_d4)

			self.affine_net_d2 = Affine_Generator('affine_net_d2', self.is_training)
			self.warped_RGB_d2, label2, _ = apply_affine_trans(self.RGB_d2, dtheta_d4)
			self.warped_RGB_d2 = tf.multiply(self.warped_RGB_d2, tf.tile(label2, [1, 1, 1, 3]))
			dtheta_d2 = self.affine_net_d2.field_model(self.warped_RGB_d2, self.IR_d2)

			self.affine_net = Affine_Generator('affine_net', self.is_training)
			self.warped_RGB_d1, label1, _ = apply_affine_trans(self.RGB, multi_affine(dtheta_d4, dtheta_d2))
			self.warped_RGB_d1 = tf.multiply(self.warped_RGB_d1, tf.tile(label1, [1, 1, 1, 3]))
			dtheta_d1 = self.affine_net.field_model(self.warped_RGB_d1, self.IR)

			self.dtheta = multi_affine(multi_affine(dtheta_d4, dtheta_d2), dtheta_d1)

			self.warped_RGB, self.label, self.resampling_grid = apply_affine_trans(self.RGB, self.dtheta)
			self.warped_RGB = tf.multiply(self.warped_RGB, tf.tile(self.label, [1, 1, 1, 3]))

			# self.fuse_before = (self.RGB + tf.tile(self.IR, [1, 1, 1, 3]))/2
			# self.fuse_after = (self.warped_RGB + tf.tile(self.IR, [1, 1, 1, 3]))/2

		if self.is_training:
			with tf.device('/gpu:1'):
				self.extract_model = Des_Extract_Model(self.batchsize, self.INPUT_H, self.INPUT_W, is_training=False, equivariance=False)
				self.extract_model.des(self.RGB, self.IR)
				self.RGB_des = self.extract_model.RGB_des
				self.IR_des = self.extract_model.IR_des
				self.RGB_des, self.IR_des = normalize_common_tf(self.RGB_des, self.IR_des)

			self.warped_RGB_des, _, _, _ = grid_sample(self.RGB_des, self.resampling_grid)
			self.w_des1 = tf.multiply(self.warped_RGB_des, self.label)
			self.w_des2 = tf.multiply(self.IR_des, self.label)

			_, label_d4, resampling_grid_d4 = apply_affine_trans(self.RGB, dtheta_d4)
			self.warped_RGB_des_d4, _, _, _ = grid_sample(self.RGB_des, resampling_grid_d4)
			self.w_des1_d4 = tf.multiply(self.warped_RGB_des_d4, label_d4)
			self.w_des2_d4 = tf.multiply(self.IR_des, label_d4)

			self.NCC_source_loss = - NCC(0.299 * self.warped_RGB[:, :, :, 0:1] + 0.587 * self.warped_RGB[:, :, :, 1:2]
										 + 0.114 * self.warped_RGB[:, :, :, 2:3], self.IR)
			self.NCC_des_loss = - NCC(self.w_des1, self.w_des2)
			self.loss = - NCC(self.w_des1, self.w_des2) - NCC(self.w_des1_d4, self.w_des2_d4)


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

class Affine_Generator(object):
	def __init__(self, scope_name, is_training):
		self.scope = scope_name
		self.is_training = is_training

	def field_model(self, img_a, img_b):
		with tf.variable_scope(self.scope):
			x = tf.concat([img_a, img_b], axis=-1)

			with tf.variable_scope('layer1'):
				weights = tf.get_variable("w1", [9, 9, 4, 16],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b1", [16], initializer=tf.constant_initializer(0.0))
				conv1 = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
				conv1 = lrelu(conv1)
				conv1_offset = convoffset2D(conv1, channel=16, scope_name='conv1_offset')
				bias = tf.get_variable("offset_b1", [16], initializer=tf.constant_initializer(0.0))
				conv1_offset = lrelu(conv1_offset + bias)
				size1 = conv1_offset.shape
				conv1_pool = tf.nn.max_pool(conv1, ksize=[1, pool_size, pool_size, 1],
											strides=[1, pool_size, pool_size, 1], padding='SAME')

			with tf.variable_scope('layer2'):
				weights = tf.get_variable("w2", [9, 9, 16, 32],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b2", [32], initializer=tf.constant_initializer(0.0))
				conv2 = tf.nn.conv2d(conv1_pool, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
				conv2 = lrelu(conv2)
				conv2_offset = convoffset2D(conv2, channel=32, scope_name='conv2_offset')
				bias = tf.get_variable("offset_b2", [32], initializer=tf.constant_initializer(0.0))
				conv2_offset = lrelu(conv2_offset + bias)
				size2 = conv2_offset.shape
				conv2_pool = tf.nn.max_pool(conv2, ksize=[1, pool_size, pool_size, 1],
											strides=[1, pool_size, pool_size, 1], padding='SAME')

			with tf.variable_scope('layer3'):
				weights = tf.get_variable("w3", [9, 9, 32, 64],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b3", [64], initializer=tf.constant_initializer(0.0))
				conv3 = tf.nn.conv2d(conv2_pool, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
				conv3 = lrelu(conv3)
				conv3_offset = convoffset2D(conv3, channel=64, scope_name='conv3_offset')
				bias = tf.get_variable("offset_b3", [64], initializer=tf.constant_initializer(0.0))
				conv3_offset = lrelu(conv3_offset + bias)
				size3 = conv3_offset.shape
				conv3_pool = tf.nn.max_pool(conv3, ksize=[1, pool_size, pool_size, 1],
											strides=[1, pool_size, pool_size, 1], padding='SAME')

			with tf.variable_scope('layer4'):
				weights = tf.get_variable("w4", [7, 7, 64, 64],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b4", [64], initializer=tf.constant_initializer(0.0))
				conv4 = tf.nn.conv2d(conv3_pool, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
				conv4 = lrelu(conv4)

			with tf.variable_scope('layer5'):
				weights = tf.get_variable("w5", [7, 7, 64, 128],
										  initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
				bias = tf.get_variable("b4", [128], initializer=tf.constant_initializer(0.0))
				conv5 = tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
				conv5 = lrelu(conv5)

			gap = tf.reduce_mean(conv5, [1, 2])

			with tf.variable_scope('fcn1'):
				gap = tf.layers.dense(inputs=gap, units=6, activation=None, use_bias=True)
				output = lrelu(gap)
		return output


def apply_affine_trans(img, dtheta):
	"img is a RGB image"
	batchsize = img.shape[0]
	oh = img.shape[1]
	ow = img.shape[2]
	x = tf.linspace(-1.0, 1.0, ow)
	y = tf.linspace(-1.0, 1.0, oh)

	identity_theta = tf.constant(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
	identity_theta = tf.expand_dims(identity_theta, axis=0)
	identity_theta = tf.tile(identity_theta, [batchsize, 1])

	affine_matrix = tf.cast(identity_theta + dtheta, 'float32')
	num_batch = batchsize
	affine_matrix = tf.reshape(affine_matrix, [num_batch, 2, 3])

	x_t, y_t = tf.meshgrid(x, y)
	# flatten
	x_t_flat = tf.reshape(x_t, [-1])
	y_t_flat = tf.reshape(y_t, [-1])

	# reshape to [x_t, y_t , 1] - (homogeneous form)
	ones = tf.ones_like(x_t_flat)
	sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])
	sampling_grid = tf.expand_dims(sampling_grid, axis=0)
	sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))
	sampling_grid = tf.cast(sampling_grid, 'float32')

	# transform the sampling grid - batch multiply
	batch_grids = tf.matmul(affine_matrix, sampling_grid)
	batch_grids = tf.reshape(batch_grids, [num_batch, 2, int(oh), int(ow)])
	resampling_grid = tf.transpose(batch_grids, (0, 2, 3, 1))

	xx, yy = tf.meshgrid(y, x)
	xx = tf.transpose(xx)
	yy = tf.transpose(yy)
	xx = tf.expand_dims(xx, -1)
	yy = tf.expand_dims(yy, -1)
	xx = tf.expand_dims(xx, 0)
	yy = tf.expand_dims(yy, 0)
	identity = tf.concat([yy, xx], axis=-1)
	defor = resampling_grid - identity

	if batchsize == 1:
		warped_R, _, _, label = grid_sample(img[0:1, :, :, 0:1], resampling_grid)
		warped_G, _, _, _ = grid_sample(img[0:1, :, :, 1:2], resampling_grid)
		warped_B, _, _, _ = grid_sample(img[0:1, :, :, 2:3], resampling_grid)
	else:
		warped_R, _, _, label = grid_sample(img[:, :, :, 0:1], resampling_grid)
		warped_G, _, _, label = grid_sample(img[:, :, :, 1:2], resampling_grid)
		warped_B, _, _, label = grid_sample(img[:, :, :, 2:3], resampling_grid)
	warped_img = tf.concat([warped_R, warped_G, warped_B], axis=-1)
	return warped_img, label, resampling_grid


def img_resize(x, scale):
	oh = x.shape[1]
	ow = x.shape[2]
	x = tf.image.resize_images(images=x, size=[tf.constant(int(oh) * scale), tf.constant(int(ow) * scale)])
	return x


def multi_affine(dtheta1, dtheta2):
	batchsize = dtheta1.shape[0]
	identity_theta = tf.constant(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
	identity_theta = tf.expand_dims(identity_theta, axis=0)
	identity_theta = tf.tile(identity_theta, [batchsize, 1])

	zeros = tf.constant(np.array([0, 0, 0], dtype=np.float32))
	zeros = tf.expand_dims(tf.expand_dims(zeros, axis=0), axis=0)
	zeros = tf.tile(zeros, [batchsize, 1, 1])
	affine_matrix1 = tf.cast(identity_theta + dtheta1, 'float32')
	affine_matrix1 = tf.reshape(affine_matrix1, [batchsize, 2, 3])
	affine_matrix1 = tf.concat([affine_matrix1, zeros], axis=1)
	affine_matrix2 = tf.cast(identity_theta + dtheta2, 'float32')
	affine_matrix2 = tf.reshape(affine_matrix2, [batchsize, 2, 3])
	affine_matrix2 = tf.concat([affine_matrix2, zeros], axis=1)

	matrix = tf.matmul(affine_matrix2, affine_matrix1)

	return tf.reshape(matrix[:, 0:2, :], [batchsize, 6]) - identity_theta

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv',
		 reuse=False):
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
							 kernel_size=kernel, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
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
		x_offset = tf_batch_map_offsets(x, offsets)
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

	for c in range(channels):
		x = input[:, :, :, c:c + 1]
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
	h=img1.shape[1]
	w=img2.shape[2]
	mean1 = tf.reduce_mean(img1, axis=[1,2])
	mean2 = tf.reduce_mean(img2, axis=[1,2])
	mean1=tf.expand_dims(mean1,axis=1)
	mean1 = tf.expand_dims(mean1, axis = 2)
	mean2=tf.expand_dims(mean2,axis=1)
	mean2 = tf.expand_dims(mean2, axis = 2)
	mean1 = tf.tile(mean1, [1, int(h), int(w), 1])
	mean2 = tf.tile(mean2, [1, int(h), int(w), 1])
	dimg1=img1-mean1
	dimg2=img2-mean2
	numerator = tf.multiply(dimg1, dimg2)

	std1 = reduce_std(img1, axis=[1,2], keepdims=True)
	std1 = tf.tile(std1, [1, int(h), int(w), 1])
	std2 = reduce_std(img2, axis = [1, 2], keepdims = True)
	std2 = tf.tile(std2, [1, int(h), int(w), 1])
	denominator = tf.multiply(std1, std2)
	return tf.reduce_mean(tf.divide(numerator, denominator))

def reduce_std(x, axis=None, keepdims=False):
	return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
	m = tf.reduce_mean(x, axis=axis, keep_dims=True)
	devs_squared = tf.square(x - m)
	return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)