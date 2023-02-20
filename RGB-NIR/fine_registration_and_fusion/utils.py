from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import cv2

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
	print('not_initialized_vars:')
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


def relu(x):
	return tf.maximum(x, 0)  # tf.nn.leaky_relu(x, alpha)


def blur(input):
	filter = tf.reshape(tf.constant([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]), [3, 3, 1, 1])
	d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
	return d/9.0

def create_defor_field(x, type=None):
	b, h, w, c = x.shape
	if type == 'non_rigid':
		output = create_non_rigid_defor_field(x)
	elif type == 'rigid':
		output = create_rigid_defor_field(x)
	elif type == 'mix':
		output1 = create_non_rigid_defor_field(x)
		output2 = create_rigid_defor_field(x)
		choose = np.random.randint(2, size=(b))
		output = np.zeros_like(output1)
		for B in range(b):
			output[B:B+1, :, :, :] = choose[B] * output1[B:B+1, :, :, :] + (1-choose[B])* output2[B:B+1, :, :, :]
	else:
		output = np.zeros([b, h, w, 2], dtype=np.float32)
	return output


def create_rigid_defor_field(x, max_defor_num=6/np.sqrt(2)):
	b, h, w, c = x.shape
	ratio = max_defor_num * 1.0 / (np.max([h, w]) / 2.0)
	dtheta = np.random.uniform(- ratio, ratio, [b, 6])

	x = np.linspace(-1.0, 1.0, w)
	y = np.linspace(-1.0, 1.0, h)
	x_t, y_t = np.meshgrid(x, y)
	# flatten
	x_t_flat = np.reshape(x_t, [-1])
	y_t_flat = np.reshape(y_t, [-1])

	# reshape to [x_t, y_t , 1] - (homogeneous form)
	ones = np.ones_like(x_t_flat)
	sampling_grid = np.stack([x_t_flat, y_t_flat, ones])
	sampling_grid = np.expand_dims(sampling_grid, axis=0)
	while sampling_grid.shape[0] < b:
		sampling_grid = np.concatenate([sampling_grid, sampling_grid], axis=0)
	sampling_grid = sampling_grid.astype(np.float32)

	identity_theta = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
	identity_theta = np.expand_dims(identity_theta, axis=0)
	while identity_theta.shape[0] < b:
		identity_theta = np.concatenate([identity_theta, identity_theta], axis=0)
	affine_theta = identity_theta + dtheta

	affine_theta = affine_theta.astype(np.float32)
	affine_theta = np.reshape(affine_theta, [b, 2, 3])

	# transform the sampling grid - batch multiply
	batch_grids = np.matmul(affine_theta, sampling_grid)
	while batch_grids.shape[0] < b:
		batch_grids = np.concatenate([batch_grids, batch_grids], axis=0)

	# print("batch grid shape: ", batch_grids.shape)
	batch_grids = np.reshape(batch_grids, [b, 2, h, w])
	output = np.transpose(batch_grids, (0, 2, 3, 1))

	xx, yy = np.meshgrid(y, x)  #
	xx = np.transpose(xx)
	yy = np.transpose(yy)
	xx = np.expand_dims(xx, -1)
	yy = np.expand_dims(yy, -1)
	xx = np.expand_dims(xx, 0)
	yy = np.expand_dims(yy, 0)
	identity = np.concatenate([yy, xx], axis=-1)
	output = output - identity
	return output



def create_non_rigid_defor_field(x): #, max_defor_num=150/np.sqrt(2)):
	b, h, w, c = x.shape
	N = 15
	ratio = 0.3 # max_defor_num * 1.0 / (np.max([h, w]) / 2.0)
	for batch in range(b):
		deforg_x = np.random.uniform(-ratio, ratio, [h, w])
		deforg_y = np.random.uniform(-ratio, ratio, [h, w])
		deforg_x_blur = cv2.blur(deforg_x, (N, N))
		deforg_y_blur = cv2.blur(deforg_y, (N, N))
		for c in range(30):
			deforg_x_blur = cv2.GaussianBlur(deforg_x_blur, (N, N), 0)
			deforg_y_blur = cv2.GaussianBlur(deforg_y_blur, (N, N), 0)
		deforg_x_blur = np.expand_dims(np.expand_dims(deforg_x_blur, axis=-1), axis=0)
		deforg_y_blur = np.expand_dims(np.expand_dims(deforg_y_blur, axis=-1), axis=0)
		deforg = np.concatenate([deforg_x_blur, deforg_y_blur], axis=-1)
		if batch == 0:
			output = deforg
		else:
			output = np.concatenate([output, deforg], axis=0)
	return output



def cmin(x):
	return np.max([x,0])

def cmax(x, patchsize):
	return np.min([x, patchsize])

def defor_reverse(defor, half_window=5):
	batchsize, height, width, _ = defor.shape
	x = np.linspace(-1.0, 1.0, height)
	y = np.linspace(-1.0, 1.0, width)
	xx, yy = np.meshgrid(y, x)
	xx = np.transpose(xx)
	yy = np.transpose(yy)
	xx = np.expand_dims(xx, -1)
	yy = np.expand_dims(yy, -1)
	xx = np.expand_dims(xx, 0)
	yy = np.expand_dims(yy, 0)
	identity = np.concatenate([yy, xx], axis=-1)
	resampling_grid = identity + defor

	defor_re = np.zeros_like(defor)

	for h_index in range(height):
		for w_index in range(width):
			y_min = cmin(h_index - half_window)
			y_max = cmax(h_index + half_window, height)
			x_min = cmin(w_index - half_window)
			x_max = cmax(w_index + half_window, width)
			diff = np.square(resampling_grid[0:batchsize, y_min:y_max, x_min:x_max, 0] - np.tile(identity[0:batchsize, h_index, w_index, 0],
					(1, y_max-y_min, x_max-x_min))) + np.square(resampling_grid[0:batchsize, y_min:y_max, x_min:x_max, 1] -
					np.tile(identity[0:batchsize, h_index, w_index, 1], (1, y_max-y_min, x_max-x_min)))
			for b in range(batchsize):
				index = diff[b, :, :].argmin()
				re_h = index // (x_max-x_min) + y_min
				re_w = index - index // (x_max-x_min) * (x_max - x_min) +x_min
				defor_re[b, h_index, w_index, 0] = - defor[b, re_h, re_w, 0]
				defor_re[b, h_index, w_index, 1] = - defor[b, re_h, re_w, 1]
	return defor_re


def rgb2ycbcr(img_rgb):
	b, _, _, _ = img_rgb.shape
	R = tf.expand_dims(img_rgb[0:b, :, :, 0], axis=-1)
	G = tf.expand_dims(img_rgb[0:b, :, :, 1], axis=-1)
	B = tf.expand_dims(img_rgb[0:b, :, :, 2], axis=-1)
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
	img_ycbcr = tf.concat([Y, Cb, Cr], axis=-1)
	return img_ycbcr


def rgb2gray(img):
	return 0.299 * img[:, :, :, 0:1] + 0.587 * img[:, :, :, 1:2] + 0.114 * img[:, :, :, 2:3]

def saturation_adjust(img, ratio):
	s = img.shape
	batchsize = int(s[0])
	dim = int(s[1])
	rgb_max = tf.reduce_max(img, axis=-1)
	rgb_min = tf.reduce_min(img, axis=-1)
	saturation = tf.reduce_mean((rgb_max - rgb_min) / (rgb_max), axis=[1, 2])
	saturation_ratio = tf.reshape((1 - saturation) * ratio, [batchsize, 1, 1, 1])
	img_gray = rgb2gray(img)
	mask = tf.clip_by_value(img - tf.tile(img_gray, [1, 1, 1, 3]), 1e-8, 1.0)
	gray_mask = (1 - rgb2gray(mask)) * tf.tile(saturation_ratio, [1, dim, dim, 1])
	img_sat = tf.clip_by_value(img * (1 + gray_mask) - tf.tile(img_gray * gray_mask, [1, 1, 1, 3]), 0, 1.0)
	return img_sat

def ycbcr2rgb(img_ycbcr):
	b, _, _, _ = img_ycbcr.shape
	Y = tf.expand_dims(img_ycbcr[0:b, :, :, 0], axis=-1)
	Cb = tf.expand_dims(img_ycbcr[0:b, :, :, 1], axis=-1)
	Cr = tf.expand_dims(img_ycbcr[0:b, :, :, 2], axis=-1)
	R = Y + 1.402 * (Cr - 128 / 255.0)
	G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
	B = Y + 1.772 * (Cb - 128 / 255.0)
	img_rgb = tf.concat([R, G, B], axis=-1)
	return img_rgb


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
	B = int(in_shape[0])
	IH = int(in_shape[1])
	IW = int(in_shape[2])


	# out_shape = grid.shape
	# OH = out_shape[1]
	# OW = out_shape[2]

	nor_ix = grid[0:B, :, :, 0]
	nor_iy = grid[0:B, :, :, 1]
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
	B = tf.shape(img)[0]
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
	wa = tf.expand_dims(w[0:B,:,:,0], axis=-1)
	wb = tf.expand_dims(w[0:B, :, :, 1], axis = -1)
	wc = tf.expand_dims(w[0:B, :, :, 2], axis = -1)
	wd = tf.expand_dims(w[0:B, :, :, 3], axis = -1)

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
		warped_R, _, _, label = grid_sample(img[0:batchsize, :, :, 0:1], resampling_grid)
		warped_G, _, _, label = grid_sample(img[0:batchsize, :, :, 1:2], resampling_grid)
		warped_B, _, _, label = grid_sample(img[0:batchsize, :, :, 2:3], resampling_grid)
	warped_img = tf.concat([warped_R, warped_G, warped_B], axis=-1)
	return warped_img, label, resampling_grid







def rgb2y_np(img_rgb):
	R = img_rgb[:, :, 0]
	G = img_rgb[:, :, 1]
	B = img_rgb[:, :, 2]
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	return Y


def affinity(F):
	batchsize = F.shape[0]
	c = F.shape[3]
	F = tf.image.resize(images=F, size=(64, 64))
	RF = tf.reshape(F,[batchsize, 64*64, c])
	RF_m = tf.matmul(RF, tf.transpose(RF, [0, 2, 1]))
	s_RF_m = tf.expand_dims(tf.nn.softmax(RF_m), axis=-1)
	return s_RF_m


def l2(x, y):
	return tf.reduce_mean(tf.square(x-y))

def l1(x, y):
	return tf.reduce_mean(tf.abs(x-y))


def l0(x, e=0.1):
	y = tf.where(tf.abs(x) > tf.ones_like(x, dtype=tf.float32) * e, tf.ones_like(x, dtype=tf.float32), tf.square(x)/(e*e))
	return tf.reduce_mean(y)

def l_small(x):
	y1 = tf.where(tf.abs(x) < tf.ones_like(x, dtype=tf.float32) * 0.5, tf.zeros_like(x, dtype=tf.float32), tf.ones_like(x, dtype=tf.float32))
	y2 = tf.where(tf.abs(x) > tf.ones_like(x, dtype=tf.float32) * 0.1, tf.ones_like(x, dtype=tf.float32), tf.zeros_like(x, dtype=tf.float32))
	y = tf.concat([y1, y2], axis=-1)
	y = tf.reduce_min(y, axis=-1)
	return tf.reduce_mean(y)

def l_large(x, e=0.2):
	y = tf.where(tf.abs(x) > tf.ones_like(x, dtype=tf.float32) * e, tf.ones_like(x, dtype=tf.float32), tf.zeros_like(x, dtype=tf.float32))
	return tf.reduce_mean(y)

def gradient(input):
	filter = tf.reshape(tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]), [3, 3, 1, 1])
	d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
	return d

def gradient_1(input):
	filter = tf.reshape(tf.constant([[0., 1., 0.], [1., -2., 0.], [0., 0., 0.]]), [3, 3, 1, 1])
	d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
	return d

def channelattention(x, y, scope='ca'):
	shape = x.shape
	channels = int(shape[-1])
	grad_x = tf.abs(gradient_channels(x))
	grad_y = tf.abs(gradient_channels(y))

	avg_p_x = tf.reduce_mean(grad_x, axis=[1, 2])
	avg_p_y = tf.reduce_mean(grad_y, axis=[1, 2])
	max_p_x = tf.reduce_max(grad_x, axis=[1, 2])
	max_p_y = tf.reduce_max(grad_y, axis=[1, 2])

	avg_p = avg_p_x + avg_p_y
	max_p = max_p_x + max_p_y
	avg_p = tf.layers.dense(avg_p, units=channels // 8,
							kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
							kernel_regularizer=None,
							use_bias=True)
	avg_p = tf.layers.dense(avg_p, units=channels,
							kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
							kernel_regularizer=None,
							use_bias=True)
	max_p = tf.layers.dense(max_p, units=channels // 8,
							kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
							kernel_regularizer=None,
							use_bias=True)
	max_p = tf.layers.dense(max_p, units=channels,
							kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
							kernel_regularizer=None,
							use_bias=True)

	scale = tf.sigmoid(avg_p + max_p, 'sigmoid')
	scale = tf.reshape(scale, [x.shape[0], 1, 1, channels])
	scale = tf.tile(scale, [1, x.shape[1], x.shape[2], 1])
	return x * scale, y * scale


def gradient_channels(input):
	shape = input.shape
	channels = int(shape[-1])
	filter=tf.reshape(tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]),[3,3,1,1])
	filter=tf.tile(filter, [1, 1, channels, 1])
	d = tf.nn.depthwise_conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
	return d


def smoothness_loss(i, mask, alpha=0.5):
	diff_1 = tf.abs(i[:, :, 1:, :] - i[:, :, :-1, :])
	diff_2 = tf.abs(i[:, 1:, :, :] - i[:, :-1, :, :])
	diff_3 = tf.abs(i[:, :-1, :-1, :] - i[:, 1:, 1:, :])
	diff_4 = tf.abs(i[:, :-1, 1:, :] - i[:, 1:, :-1, :])
	if i is not None and alpha > 0.0:
		w1 = tf.exp(-alpha * tf.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]))
		w2 = tf.exp(- alpha * tf.abs(mask[:, 1:, :, :] - mask[:, :-1, :, :]))
		w3 = tf.exp(- alpha * tf.abs(mask[:, :-1, :-1, :] - mask[:, 1:, 1:, :]))
		w4 = tf.exp(- alpha * tf.abs(mask[:, :-1, :1, :] - mask[:, 1:, :-1, :]))
	else:
		w1 = w2 = w3 = w4 = 1.0
	loss = tf.reduce_mean(w1 * diff_1) + tf.reduce_mean(w2 * diff_2) + tf.reduce_mean(
		w3 * diff_3) + tf.reduce_mean(w4 * diff_4)
	return loss



def SSIM(img1, img2, size = 11, sigma = 1.5):
	window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
	k1 = 0.01
	k2 = 0.03
	L = 1  # depth of image (255 in case the image has a different scale)
	c1 = (k1 * L) ** 2
	c2 = (k2 * L) ** 2
	mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu1_sq = mu1 * mu1
	mu2_sq = mu2 * mu2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_sq
	sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
	sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

	# value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
	ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
	value = tf.reduce_mean(ssim_map)
	return value



def _tf_fspecial_gauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function
	"""
	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

	x_data = np.expand_dims(x_data, axis = -1)
	x_data = np.expand_dims(x_data, axis = -1)

	y_data = np.expand_dims(y_data, axis = -1)
	y_data = np.expand_dims(y_data, axis = -1)

	x = tf.constant(x_data, dtype = tf.float32)
	y = tf.constant(y_data, dtype = tf.float32)

	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
	return g / tf.reduce_sum(g)


def SD(x):
	b, h, w, c = x.shape
	mu = tf.reshape(tf.reduce_mean(x, axis=[1, 2, 3]), [b, 1, 1, 1])
	mu = tf.tile(mu, [1, h, w, c])
	return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(x-mu), axis=[1, 2, 3])))
