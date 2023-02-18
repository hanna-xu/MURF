from __future__ import print_function
import tensorflow as tf
import os
import numpy as np

def listdir(path):
	list_name=[]
	for file in os.listdir(path):
		file_path = os.path.join(path, file)
		if os.path.isdir(file_path):
			os.listdir(file_path, list_name)
		else:
			list_name.append(file_path)
	return list_name


def initialize_uninitialized(sess):
	global_vars = tf.compat.v1.global_variables()
	is_not_initialized = sess.run([tf.compat.v1.is_variable_initialized(var) for var in global_vars])
	not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
	# print('not_initialized_vars:')
	# for i in not_initialized_vars:
	# 	print(str(i.name))
	if len(not_initialized_vars):
		sess.run(tf.compat.v1.variables_initializer(not_initialized_vars))


def load(sess, saver, checkpoint_dir):
	print(" [*] Reading checkpoints...")
	# checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	print("ckpt: ", ckpt)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
		counter = ckpt_name.split('-')[-1]
		print("counter: ", counter)
		counter = int(counter.split('.')[0])
		print(" [*] Success to read {}".format(ckpt_name))
		return True, counter
	else:
		print(" [*] Failed to find a checkpoint")
		return False, 0





def up_layer(scope_name, x, channels, kernel_size):
	with tf.variable_scope(scope_name):
	# 	x = tf.contrib.layers.conv2d_transpose(x, num_outputs=channels * 2, kernel_size=5, stride = 2, padding ='SAME')
		l=int((kernel_size-1)/2)

		_, height, width, c = x.shape
		x = tf.image.resize_images(images=x, size=[tf.constant(int(height) * 2), tf.constant(int(width) * 2)])


		x = tf.pad(x, [[0, 0], [l, l], [l, l], [0, 0]], mode='REFLECT')
		x = tf.contrib.layers.conv2d(inputs=x, num_outputs=channels, kernel_size=kernel_size, stride = 1, padding='VALID',
									 activation_fn=None)
		x = lrelu(x)
	return x

def lrelu(x, alpha=0.2):
	return tf.maximum(x, alpha * x)  # tf.nn.leaky_relu(x, alpha)


def rgb2ycbcr(img_rgb):
	R = tf.expand_dims(img_rgb[:, :, :, 0], axis=-1)
	G = tf.expand_dims(img_rgb[:, :, :, 1], axis=-1)
	B = tf.expand_dims(img_rgb[:, :, :, 2], axis=-1)
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
	img_ycbcr = tf.concat([Y, Cb, Cr], axis=-1)
	return img_ycbcr

def normalize_common_tf(a, b):
	# a_min, a_max = np.percentile(a, [1, 99])
	# b_min, b_max = np.percentile(b, [1, 99])
	a_min = tf.reduce_min(a, axis=[1, 2, 3])
	a_max = tf.reduce_max(a, axis=[1, 2, 3])
	b_min = tf.reduce_min(b, axis=[1, 2, 3])
	b_max = tf.reduce_max(b, axis=[1, 2, 3])
	minimum = tf.reduce_min(tf.concat([a_min, b_min], axis=-1))
	maximum = tf.reduce_max(tf.concat([a_max, b_max], axis=-1))
	new_a = tf.clip_by_value((a - minimum) / (maximum - minimum), 1e-10, 1.0)
	new_b = tf.clip_by_value((b - minimum) / (maximum - minimum), 1e-10, 1.0)
	return new_a, new_b


def resblock(x_init, channels, use_bias=True, scope='resblock', reuse=False):
	with tf.variable_scope(scope):
		with tf.variable_scope('res1'):
			x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias,
					 reuse=reuse)
			# x = instance_norm(x)
			x = lrelu(x)

		with tf.variable_scope('res2'):
			x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias,
					 reuse=reuse)
			# x = instance_norm(x)

		return x + x_init


def grid_sample(input, grid):
	in_shape = input.shape
	IH = int(in_shape[1])
	IW = int(in_shape[2])


	# out_shape = grid.shape
	# OH = out_shape[1]
	# OW = out_shape[2]

	nor_ix = grid[:, :, :, 0]
	nor_iy = grid[:, :, :, 1]
	ix = ((nor_ix + 1) / 2) * (IW - 1)
	iy = ((nor_iy + 1) / 2) * (IH - 1)
	out, label = bilinear_sampler(input, ix, iy)
	return out, ix, iy, label


def bilinear_sampler(img, x, y):
	"""
	Performs bilinear sampling of the input images according to the
	normalized coordinates provided by the sampling grid. Note that
	the sampling is done identically for each channel of the input.
	To test if the function works properly, output image should be
	identical to input image when theta is initialized to identity
	transform.
	Input
	-----
	- img: batch of images in (B, H, W, C) layout.
	- grid: x, y which is the output of affine_grid_generator.
	Returns
	-------
	- out: interpolated images according to grids. Same size as grid.
	"""
	H = tf.shape(img)[1]
	W = tf.shape(img)[2]
	max_y = tf.cast(H - 1, 'int32')
	max_x = tf.cast(W - 1, 'int32')
	zero = tf.zeros([], dtype='int32')

	# rescale x and y to [0, W-1/H-1]
	# x = tf.cast(x, 'float32')
	# y = tf.cast(y, 'float32')
	# x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
	# y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

	# grab 4 nearest corner points for each (x_i, y_i)
	x0 = tf.cast(tf.floor(x), 'int32')
	x1 = x0 + 1
	y0 = tf.cast(tf.floor(y), 'int32')
	y1 = y0 + 1

	labelx_min = tf.where(x < tf.cast(zero, 'float32'), x = tf.zeros_like(x), y = tf.ones_like(x))
	labelx_max = tf.where(x > tf.cast(max_x, 'float32'), x = tf.zeros_like(x), y = tf.ones_like(x))
	labelx = tf.multiply(labelx_min, labelx_max)
	labely_min = tf.where(y < tf.cast(zero, 'float32'), x = tf.zeros_like(y), y = tf.ones_like(y))
	labely_max = tf.where(y > tf.cast(max_y, 'float32'), x = tf.zeros_like(y), y = tf.ones_like(y))
	labely = tf.multiply(labely_min, labely_max)
	label=tf.expand_dims(tf.multiply(labelx, labely), axis=-1)

	# clip to range [0, H-1/W-1] to not violate img boundaries
	x0 = tf.clip_by_value(x0, zero, max_x)
	x1 = tf.clip_by_value(x1, zero, max_x)
	y0 = tf.clip_by_value(y0, zero, max_y)
	y1 = tf.clip_by_value(y1, zero, max_y)

	# get pixel value at corner coords
	Ia = get_pixel_value(img, x0, y0)
	Ib = get_pixel_value(img, x0, y1)
	Ic = get_pixel_value(img, x1, y0)
	Id = get_pixel_value(img, x1, y1)

	# recast as float for delta calculation
	x0 = tf.cast(x0, 'float32')
	x1 = tf.cast(x1, 'float32')
	y0 = tf.cast(y0, 'float32')
	y1 = tf.cast(y1, 'float32')

	# calculate deltas
	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)

	# add dimension for addition
	wa = tf.expand_dims(wa, axis=3)
	wb = tf.expand_dims(wb, axis=3)
	wc = tf.expand_dims(wc, axis=3)
	wd = tf.expand_dims(wd, axis=3)
	w = tf.concat([wa, wb, wc, wd], axis=-1)
	w = tf.nn.softmax(w/0.05)
	wa = tf.expand_dims(w[:,:,:,0], axis=-1)
	wb = tf.expand_dims(w[:, :, :, 1], axis = -1)
	wc = tf.expand_dims(w[:, :, :, 2], axis = -1)
	wd = tf.expand_dims(w[:, :, :, 3], axis = -1)

	# compute output
	out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
	out = tf.multiply(out, label)
	return out, label


def get_pixel_value(img, x, y):
	"""
	Utility function to get pixel value for coordinate
	vectors x and y from a  4D tensor image.
	Input
	-----
	- img: tensor of shape (B, H, W, C)
	- x: flattened tensor of shape (B*H*W,)
	- y: flattened tensor of shape (B*H*W,)
	Returns
	-------
	- output: tensor of shape (B, H, W, C)
	"""
	shape = tf.shape(x)
	batch_size = shape[0]
	height = shape[1]
	width = shape[2]

	batch_idx = tf.range(0, batch_size)
	batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
	b = tf.tile(batch_idx, (1, height, width))

	indices = tf.stack([b, y, x], 3)

	return tf.gather_nd(img, indices)

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


def rgb2y_np(img_rgb):
	R = img_rgb[:, :, 0]
	G = img_rgb[:, :, 1]
	B = img_rgb[:, :, 2]
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	return Y