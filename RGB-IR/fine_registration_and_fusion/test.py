from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
from scipy.io import loadmat
from f2m_model import F2M_Model
from utils import *
import scipy.misc
import cv2
from datetime import datetime
import scipy.io as scio

N = 512
def main():
	test_path1 = './test_imgs/RGB/'
	test_path2 = './test_imgs/IR/'
	save_path = './results/'
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	if not os.path.exists(save_path + 'fused_img/'):
		os.mkdir(save_path + 'fused_img/')
	files = listdir(test_path1)
	if not os.path.exists(save_path + 'compare/'):
		os.mkdir(save_path + 'compare/')
	files = listdir(test_path1)

	pic_num = 0
	T=[]

	with tf.Graph().as_default(), tf.Session() as sess:
		f2m_model = F2M_Model(BATCH_SIZE=1, INPUT_W = N, INPUT_H = N, is_training=False, BN=True)
		SOURCE_RGB = tf.placeholder(tf.float32, shape = (1, N, N, 3), name = 'SOURCE1')
		SOURCE_IR = tf.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE2')
		DEFOR_FIELD = tf.placeholder(tf.float32, shape=(1, N, N, 2), name='defor_field')
		RE_DEFOR_FIELD_GT = tf.placeholder(tf.float32, shape=(1, N, N, 2), name='re_defor_field_gt')
		f2m_model.f2m(SOURCE_RGB, SOURCE_IR, DEFOR_FIELD, RE_DEFOR_FIELD_GT)

		var_list_f2m = tf.contrib.framework.get_trainable_variables(scope='f2m_net')
		g_list = tf.global_variables()
		bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
		bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
		var_list_f2m += bn_moving_vars
		var_list_bn = list()
		var_list_f2m_fuse = list()
		for i in var_list_f2m:
			if "BatchNorm" in i.name:
				var_list_bn.append(i)
			if "offset" not in i.name:
				var_list_f2m_fuse.append(i)

		saver = tf.compat.v1.train.Saver(var_list=var_list_f2m)
		'''only for fusing aligned images'''
		# saver = tf.compat.v1.train.Saver(var_list=var_list_f2m_fuse)

		global_vars = tf.compat.v1.global_variables()
		sess.run(tf.compat.v1.variables_initializer(global_vars))
		_, _ = load(sess, saver, './models/finetuning/')

		for file in files:
			pic_num += 1
			name = file.split('/')[-1]
			name = name.split('.')[-2]
			print("\033[0;33;40m["+ str(pic_num) + "/" + str(len(files)) +"]: "+ name + ".jpg" + "\033[0m")
			rgb_img = scipy.misc.imread(test_path1 + file.split('/')[-1])
			ir_img = scipy.misc.imread(test_path2 + file.split('/')[-1])

			start_time = datetime.now()

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

			defor_field = np.zeros([1, N, N, 2])
			defor_re = np.zeros_like(defor_field)
			FEED_DICT = {SOURCE_RGB: rgb_img_N, SOURCE_IR: ir_img_N, DEFOR_FIELD: defor_field,
						 RE_DEFOR_FIELD_GT: defor_re}

			fused_img = sess.run(f2m_model.fused_img, feed_dict=FEED_DICT)
			fused_img = scipy.misc.imresize(fused_img[0, :, :, :], (rgb_dimension[0], rgb_dimension[1])).astype(
				np.float32) / 255.0
			scipy.misc.imsave(save_path + 'fused_img/' + name + '.jpg', fused_img)

			time = datetime.now() - start_time
			time=time.total_seconds()

			if pic_num>1:
				T.append(time)
				print("\nElapsed_time: %s" % (T[pic_num-2]))

			fused_ori = (rgb_img/255.0 + np.tile(np.expand_dims(ir_img/255.0, axis=-1), [1, 1, 3]))/2
			compare = np.concatenate([fused_ori, fused_img], axis = 1)
			scipy.misc.imsave(save_path + 'compare/' + name + '.png', compare)

		print("Time mean :%s, std: %s\n" % (np.mean(T), np.std(T)))


if __name__ == '__main__':
	main()


