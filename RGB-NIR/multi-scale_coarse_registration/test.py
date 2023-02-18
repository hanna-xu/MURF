from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
from scipy.io import loadmat
from affine_model import Affine_Model, apply_affine_trans
from utils import *
import scipy.misc
import cv2
import scipy.io as scio
from datetime import datetime

H = 1024
W = 1024
def main():
	test_path1 = './test_data/images/RGB/'
	test_path2 = './test_data/images/NIR/'
	test_path_LM = './test_data/LM/'
	save_path = './results/'
	files = listdir(test_path1)
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	pic_num = 0
	Ti=[]

	with tf.Graph().as_default(), tf.Session() as sess:
		affine_model = Affine_Model(BATCH_SIZE=1, INPUT_H=H, INPUT_W=W, is_training=False)
		SOURCE_RGB_N = tf.placeholder(tf.float32, shape = (1, H, W, 3), name = 'SOURCE1_N')
		SOURCE_NIR_N = tf.placeholder(tf.float32, shape = (1, H, W, 1), name = 'SOURCE2_N')
		affine_model.affine(SOURCE_RGB_N, SOURCE_NIR_N, dropout=False)
		WSOURCE1, label, _ = apply_affine_trans(SOURCE_RGB_N, affine_model.dtheta)
		WSOURCE1 = tf.multiply(WSOURCE1, tf.tile(label, [1, 1, 1, 3]))

		var_list_affine = tf.contrib.framework.get_trainable_variables(scope='affine_net')
		var_list = var_list_affine

		saver = tf.compat.v1.train.Saver(var_list=var_list)
		global_vars = tf.compat.v1.global_variables()
		sess.run(tf.compat.v1.variables_initializer(global_vars))
		_, _ = load(sess, saver, './models_finetuning/')

		for file in files:
			pic_num += 1
			names = file.split('/')[-1]
			name = names.split('.')[-2]
			print("\033[0;33;40m[" + str(pic_num) + "/" + str(len(files)) + "]: " + names + "\033[0m")

			'''load data through image'''
			rgb_img = scipy.misc.imread(test_path1 + names)
			nir_img = scipy.misc.imread(test_path2 + names)

			'''load data with landmark through .mat'''
			# data = scio.loadmat(test_path_LM + name + '.mat')
			# rgb_img = data['I_move']
			# nir_img = data['I_fix']
			# LMmove = data['LMmove']
			# LMfix = data['LMfix']

			start_time = datetime.now()
			rgb_dimension = list(rgb_img.shape)
			nir_dimension = list(nir_img.shape)
			height = rgb_dimension[0]
			width = rgb_dimension[1]

			"resize"
			rgb_img_N = scipy.misc.imresize(rgb_img, size=(H, W))
			rgb_img_N = np.expand_dims(rgb_img_N, axis=0)
			rgb_img_N = rgb_img_N.astype(np.float32) / 255.0
			nir_img_N = scipy.misc.imresize(nir_img, size=(H, W))
			nir_img_N = np.expand_dims(np.expand_dims(nir_img_N, axis=0), axis=-1)
			nir_img_N = nir_img_N.astype(np.float32) / 255.0

			warped_rgb, dtheta = sess.run([WSOURCE1, affine_model.dtheta], feed_dict={SOURCE_RGB_N: rgb_img_N, SOURCE_NIR_N: nir_img_N})
			warped_rgb = scipy.misc.imresize(warped_rgb[0, :, :, :], (height, width)).astype(np.float32) / 255.0
			if not os.path.exists(save_path + 'warped_RGB/'):
				os.mkdir(save_path + 'warped_RGB/')
			scipy.misc.imsave(save_path + 'warped_RGB/' + name + '.png', warped_rgb)

			time = datetime.now() - start_time
			time = time.total_seconds()

			if pic_num>1:
				Ti.append(time)
				print("\nElapsed_time: %s" % (Ti[pic_num-2]))
				print("Time mean :%s, std: %s\n" % (np.mean(Ti), np.std(Ti)))

			rgb_img = rgb_img.astype(np.float32) / 255.0
			nir_img = nir_img.astype(np.float32) / 255.0
			fused_ori = (rgb_img + np.tile(np.expand_dims(nir_img, axis=-1), [1, 1, 3]))/2
			fused = (warped_rgb + np.tile(np.expand_dims(nir_img, axis=-1), [1, 1, 3]))/2
			compare = np.concatenate([fused_ori, fused], axis = 1)
			if not os.path.exists(save_path + 'compare/'):
				os.mkdir(save_path + 'compare/')
			scipy.misc.imsave(save_path + 'compare/' + name + '.png', compare)

			'''If load data with landmark'''
			'''calculate landmark after deformation '''
			# identity_theta = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
			# affine_matrix = np.reshape(identity_theta + dtheta[0, :], [2, 3])
			# print(affine_matrix)
			# Wd = W * 1.0
			# Hd = H * 1.0
			# LMmove_x = LMmove[:, 0:1] / (width - 1) * (Wd - 1) + (width - Wd) / (width - 1)
			# LMmove_y = LMmove[:, 1:2] / (height - 1) * (Hd - 1) + (height - Hd) / (height - 1)
			# LMmove_x = (LMmove_x - 1) / (Wd - 1) * 2 - 1
			# LMmove_y = (LMmove_y - 1) / (Hd - 1) * 2 - 1
			# LMmove1 = np.concatenate([LMmove_x, LMmove_y, np.ones([5, 1], dtype=np.float32)], axis=-1)
			# LMmove1 = np.transpose(LMmove1, (1, 0))
			# R = np.linalg.inv(affine_matrix[:, 0:2])
			# T = - affine_matrix[:, 2:3]
			# P = np.concatenate([R, T], axis=-1)
			# LMmove_test1 = np.matmul(P, LMmove1)
			# LMmove_test_x = (LMmove_test1[0:1, :] + 1) / 2 * (Wd - 1) + 1
			# LMmove_test_y = (LMmove_test1[1:2, :] + 1) / 2 * (Hd - 1) + 1
			# LMmove_test_x = (LMmove_test_x - (width - Wd) / (width - 1)) / (Wd - 1) * (width - 1)
			# LMmove_test_y = (LMmove_test_y - (height - Wd) / (height - 1)) / (Hd - 1) * (height - 1)
			# LMmove_test = np.concatenate([LMmove_test_x, LMmove_test_y], axis=0)
			# LMmove_test = np.transpose(LMmove_test, (1, 0))
			#
			# if not os.path.exists(save_path + 'LM/'):
			# 	os.mkdir(save_path + 'LM/')
			# scio.savemat(save_path + 'LM/' + name + '.mat',
			# 			 {'LMmove_test': LMmove_test, 'LMmove': LMmove, 'LMfix': LMfix, 'matrix':affine_matrix})
			# print('LMmove_test\n', LMmove_test)
			# print('LMfix\n', LMfix)

if __name__ == '__main__':
	main()


