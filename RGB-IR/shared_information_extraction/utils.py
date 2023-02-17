from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import math

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
	with tf.compat.v1.variable_scope(scope_name):
		l=int((kernel_size-1)/2)

		_, height, width, c = x.shape
		x = tf.image.resize(images=x, size=[tf.constant(int(height) * 2), tf.constant(int(width) * 2)])

		x = tf.pad(x, [[0, 0], [l, l], [l, l], [0, 0]], mode='REFLECT')
		x = tf.contrib.layers.conv2d(inputs=x, num_outputs=channels, kernel_size=kernel_size, stride = 1, padding='VALID',
									 activation_fn=None)
		x = lrelu(x)
	return x

def lrelu(x, alpha=0.2):
	return tf.maximum(x, alpha * x)  # tf.nn.leaky_relu(x, alpha)


def ds(x):
	_, height, width, c = x.shape
	x = tf.image.resize(images=x, size=[tf.constant(int(height//2)), tf.constant(int(width//2))])
	return x

def normalize_common(a, b):
	a_min, a_max = np.percentile(a, [1, 99])
	b_min, b_max = np.percentile(b, [1, 99])
	minimum = min(a_min, b_min)
	maximum = max(a_max, b_max)
	new_a = np.clip((a - minimum) / (maximum - minimum), 0, 1)
	new_b = np.clip((b - minimum) / (maximum - minimum), 0, 1)
	return new_a, new_b


def normalize_common_tf(a, b):
	a_min = tf.reduce_min(a, axis=[1, 2, 3])
	a_max = tf.reduce_max(a, axis=[1, 2, 3])
	b_min = tf.reduce_min(b, axis=[1, 2, 3])
	b_max = tf.reduce_max(b, axis=[1, 2, 3])
	minimum = tf.reduce_min(tf.concat([a_min, b_min], axis=-1))
	maximum = tf.reduce_max(tf.concat([a_max, b_max], axis=-1))
	new_a = tf.clip_by_value((a - minimum) / (maximum - minimum), 1e-10, 1.0)
	new_b = tf.clip_by_value((b - minimum) / (maximum - minimum), 1e-10, 1.0)
	return new_a, new_b


def affinity(F):
	batchsize = F.shape[0]
	c = F.shape[3]
	F = tf.image.resize(images=F, size=(32, 32))
	RF = tf.reshape(F,[batchsize, 32*32, c])
	RF_m = tf.matmul(RF, tf.transpose(RF, [0, 2, 1]))
	s_RF_m = tf.expand_dims(tf.nn.softmax(RF_m), axis=-1)
	return s_RF_m




def l1_loss(x,y):
	return tf.reduce_mean(tf.abs(x-y))


def grad(img):
	kernel = tf.constant([[0, 1.0, 0], [1.0, -4, 1.0], [0, 1.0, 0]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g



def mean(img):
	kernel = tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=tf.float32)
	kernel = tf.expand_dims(kernel, axis=-1)
	kernel = tf.expand_dims(kernel, axis=-1)
	g = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
	return g


def rgb2ycbcr(img_rgb):
	R = tf.expand_dims(img_rgb[:, :, :, 0], axis=-1)
	G = tf.expand_dims(img_rgb[:, :, :, 1], axis=-1)
	B = tf.expand_dims(img_rgb[:, :, :, 2], axis=-1)
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
	img_ycbcr = tf.concat([Y, Cb, Cr], axis=-1)
	return img_ycbcr


def l2_loss(x,y):
	norm = tf.reduce_mean(tf.square(x-y))
	return norm


def batch_rotate_p4(batch, degrees):
	"""Rotates by k*90 degrees each sample in a batch.
	Args:
		batch (Tensor): the batch to rotate, format is (N, C, H, W).
		k (list of int): the rotations to perform for each sample k[i]*90 degrees.
		device (str): the device to allocate memory and run computations on.

	Returns (Tensor):
		The rotated batch.
	"""
	batch_size = batch.shape[0]
	channels = batch.shape[-1]
	assert len(degrees) == batch_size, "The size of k must be equal to the batch size."

	for i in range(batch_size):
		for c in range(channels):
			rotate_c = tf.contrib.image.rotate(batch[i:i + 1, :, :, c:c+1], degrees[i] * 90 * math.pi / 180,
											 interpolation='BILINEAR')
			if c==0:
				rotate = rotate_c
			else:
				rotate = tf.concat([rotate, rotate_c], axis=-1)
		if i == 0:
			batch_rotate = rotate
		else:
			batch_rotate = tf.concat([batch_rotate, rotate], axis=0)
	return batch_rotate


def scroll(data):
	batchsize = data.shape[0]
	H = data.shape[1]
	W = data.shape[2]
	channels = data.shape[3]
	random_margin = np.random.randint(10, size=batchsize) + 5
	for i in range(batchsize):
		for c in range(2):
			if c == 0:
				scroll_d = tf.concat([data[i:i + 1, random_margin[i]:H, :, :], data[i:i + 1, 0:random_margin[i], :, :]],
									 axis=1)
			else:
				scroll_d = tf.concat([data[i:i + 1, :, random_margin[i]:W, :], data[i:i + 1, :, 0:random_margin[i], :]],
									 axis=2)
		if i == 0:
			scroll_data = scroll_d
		else:
			scroll_data = tf.concat([scroll_data, scroll_d], axis=0)
	return scroll_data
