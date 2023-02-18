from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
from scipy.io import loadmat
from affine_model import Affine_Model
from utils import *
import scipy.misc
import cv2
import scipy.io as scio

N = 512
def main():
	test_path1 = './test_data/images/RGB/'
	test_path2 = './test_data/images/IR/'
	test_path_LM = './test_data/LM/'
	save_path = './results/'
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	files = listdir(test_path1)

	pic_num = 0

	with tf.Graph().as_default(), tf.Session() as sess:
		affine_model = Affine_Model(BATCH_SIZE=1, INPUT_W=N, INPUT_H=N, is_training=True)
		SOURCE_RGB = tf.placeholder(tf.float32, shape = (1, N, N, 3), name = 'SOURCE1')
		SOURCE_IR = tf.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE2')
		affine_model.affine(SOURCE_RGB, SOURCE_IR)

		var_list_des = tf.contrib.framework.get_trainable_variables(scope='des_extract')
		var_list_affine = tf.contrib.framework.get_trainable_variables(scope='affine_net')
		var_list = var_list_affine + var_list_des
		affine_model.solver = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.99).minimize(
			affine_model.loss, var_list=var_list_affine)
		saver = tf.compat.v1.train.Saver(var_list=var_list)

		for file in files:
			pic_num += 1
			global_vars = tf.compat.v1.global_variables()
			sess.run(tf.compat.v1.variables_initializer(global_vars))
			_, _ = load(sess, saver, './models/')

			names = file.split('/')[-1]
			name = names.split('.')[-2]
			print("\033[0;37;40m\t["+ str(pic_num) + "/" + str(len(files)) +"]: "+ names + "\033[0m")

			'''load data through image'''
			# rgb_img = scipy.misc.imread(test_path1 + names)
			# ir_img = scipy.misc.imread(test_path2 + names)

			'''load data with landmark through .mat'''
			data = scio.loadmat(test_path_LM + name + '.mat')
			rgb_img = data['I_move']
			ir_img = data['I_fix']
			LMmove = data['LMmove']
			LMfix = data['LMfix']

			rgb_dimension = list(rgb_img.shape)
			ir_dimension = list(ir_img.shape)
			H = rgb_dimension[0] * 1.0
			W = rgb_dimension[1] * 1.0
			"resize"
			rgb_img_N = scipy.misc.imresize(rgb_img, size=(N, N))
			ir_img_N = scipy.misc.imresize(ir_img, size=(N, N))
			rgb_img_N = np.expand_dims(rgb_img_N, axis=0)
			ir_img_N = np.expand_dims(np.expand_dims(ir_img_N, axis=0), axis=-1)
			rgb_img_N = rgb_img_N.astype(np.float32)/255.0
			ir_img_N = ir_img_N.astype(np.float32) / 255.0

			for p in range(20):
				sess.run(affine_model.solver, feed_dict={SOURCE_RGB: rgb_img_N, SOURCE_IR: ir_img_N})
				# print(sess.run(affine_model.loss, feed_dict={SOURCE_RGB: rgb_img_N, SOURCE_IR: ir_img_N}))

			warped_rgb, dtheta = sess.run([affine_model.warped_RGB, affine_model.dtheta],
										  feed_dict={SOURCE_RGB: rgb_img_N, SOURCE_IR: ir_img_N})
			warped_rgb = scipy.misc.imresize(warped_rgb[0, :, :, :], (rgb_dimension[0], rgb_dimension[1])).astype(np.float32) / 255.0
			if not os.path.exists(save_path + 'warped_RGB/'):
				os.mkdir(save_path + 'warped_RGB/')
			scipy.misc.imsave(save_path + 'warped_RGB/' + name + '.jpg', warped_rgb)

			rgb_img = rgb_img.astype(np.float32) / 255.0
			ir_img = ir_img.astype(np.float32) / 255.0
			fused_ori = (rgb_img + np.tile(np.expand_dims(ir_img, axis=-1), [1, 1, 3]))/2
			fused = (warped_rgb + np.tile(np.expand_dims(ir_img, axis=-1), [1, 1, 3]))/2
			compare = np.concatenate([fused_ori, fused], axis = 1)
			if not os.path.exists(save_path + 'compare/'):
				os.mkdir(save_path + 'compare/')
			scipy.misc.imsave(save_path + 'compare/' + name + '.png', compare)

			'''If load data with landmark, calculate landmark after deformation '''
			identity_theta = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
			affine_matrix = np.reshape(identity_theta + dtheta[0, :], [2, 3])
			print(affine_matrix)
			LMmove_x = LMmove[:, 0:1] / (W - 1) * (N - 1) + (W - N) / (W - 1)
			LMmove_y = LMmove[:, 1:2] / (H - 1) * (N - 1) + (H - N) / (H - 1)
			LMmove_x = (LMmove_x - 1) / (N - 1) * 2 - 1
			LMmove_y = (LMmove_y - 1) / (N - 1) * 2 - 1
			LMmove1 = np.concatenate([LMmove_x, LMmove_y, np.ones([5,1], dtype=np.float32)], axis=-1)
			LMmove1 = np.transpose (LMmove1, (1, 0))
			R = np.linalg.inv(affine_matrix[:,0:2])
			T = - affine_matrix[:,2:3]
			P = np.concatenate([R, T], axis=-1)
			LMmove_test1 = np.matmul(P, LMmove1)
			LMmove_test_x = (LMmove_test1[0:1, :] + 1) / 2 * (N - 1) + 1
			LMmove_test_y = (LMmove_test1[1:2, :] + 1) / 2 * (N - 1) + 1
			LMmove_test_x = (LMmove_test_x - (W - N) / (W - 1)) / (N - 1) * (W - 1)
			LMmove_test_y = (LMmove_test_y - (H - N) / (H - 1)) / (N - 1) * (H - 1)
			LMmove_test = np.concatenate([LMmove_test_x, LMmove_test_y], axis=0)
			LMmove_test = np.transpose(LMmove_test, (1, 0))
			print('LMmove_test\n', LMmove_test)
			print('LMfix\n', LMfix)
			if not os.path.exists(save_path + 'LM/'):
				os.mkdir(save_path + 'LM/')
			scio.savemat(save_path + 'LM/' + name + '.mat',
						 {'LMmove_test': LMmove_test, 'LMmove': LMmove, 'LMfix': LMfix})

if __name__ == '__main__':
	main()


