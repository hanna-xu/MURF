from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
from scipy.io import loadmat
from train import train_descriptor
from des_extract_model import Des_Extract_Model
from utils import *
import scipy.misc
import cv2

N=512

def main():
	test_path1 = './test_imgs/RGB/'
	test_path2 = './test_imgs/IR/'
	save_path = './des_results/'
	files = listdir(test_path1)

	with tf.Graph().as_default(), tf.Session() as sess:
		model = Des_Extract_Model(BATCH_SIZE=1, INPUT_W=N, INPUT_H=N, is_training=False, equivariance=False)
		SOURCE_RGB = tf.placeholder(tf.float32, shape = (1, N, N, 3), name = 'SOURCE1')
		SOURCE_IR = tf.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE2')

		model.des(SOURCE_RGB, SOURCE_IR)
		saver = tf.compat.v1.train.Saver(max_to_keep=5)

		var_list_rgb = tf.contrib.framework.get_trainable_variables(scope='RGB_Encoder')
		var_list_ir = tf.contrib.framework.get_trainable_variables(scope='IR_Encoder')
		var_list_descriptor = var_list_rgb + var_list_ir

		initialize_uninitialized(sess)
		_, _ = load(sess, saver, './models')

		for file in files:
			name = file.split('/')[-1]
			print(name)
			rgb_img = scipy.misc.imread(test_path1 + name)
			ir_img = scipy.misc.imread(test_path2 + name)
			rgb_dimension = list(rgb_img.shape)
			ir_dimension = list(ir_img.shape)
			rgb_img = scipy.misc.imresize(rgb_img, (N, N))
			ir_img = scipy.misc.imresize(ir_img, size=(N, N))
			rgb_img = np.expand_dims(rgb_img, axis=0)
			ir_img = np.expand_dims(np.expand_dims(ir_img, axis=0), axis=-1)
			rgb_img = rgb_img.astype(np.float32) / 255.0
			ir_img = ir_img.astype(np.float32) / 255.0

			rgb_des, ir_des =  sess.run([model.RGB_des, model.IR_des], feed_dict={SOURCE_RGB: rgb_img, SOURCE_IR: ir_img})
			rgb_des, ir_des = normalize_common(rgb_des, ir_des)

			if not os.path.exists(save_path):
				os.mkdir(save_path)
			if not os.path.exists(save_path + 'RGB/'):
				os.mkdir(save_path + 'RGB/')
			if not os.path.exists(save_path + 'IR/'):
				os.mkdir(save_path + 'IR/')

			scipy.misc.imsave(save_path + 'RGB/' + name, scipy.misc.imresize(rgb_des[0, :, :, 0], (rgb_dimension[0], rgb_dimension[1])))
			scipy.misc.imsave(save_path + 'IR/' + name, scipy.misc.imresize(ir_des[0, :, :, 0], (ir_dimension[0], ir_dimension[1])))


if __name__ == '__main__':
	main()


