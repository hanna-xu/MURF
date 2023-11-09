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

N = 1024 # set it according to the resolution of source images, e.g., if the source image is of size 1240x1080, N can be set as 1240
def main():
	test_path1 = './test_imgs/RGB/'
	test_path2 = './test_imgs/NIR/'
	save_path = './results/'
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	files = listdir(test_path1)

	pic_num = 0
	T=[]

	with tf.Graph().as_default(), tf.Session() as sess:
		f2m_model = F2M_Model(BATCH_SIZE=1, INPUT_W = None, INPUT_H = None, is_training=True)
		SOURCE_RGB = tf.placeholder(tf.float32, shape = (1, N, N, 3), name = 'SOURCE1')
		SOURCE_NIR = tf.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE2')
		f2m_model.f2m(SOURCE_RGB, SOURCE_NIR, is_training=False)

		var_list_f2m = tf.contrib.framework.get_trainable_variables(scope='f2m_net')
		var_list_f2m_offset = tf.contrib.framework.get_trainable_variables(scope='f2m_net/conv1_offset')
		var_list_f2m_fuse=[]
		for var in var_list_f2m:
			if var not in var_list_f2m_offset:
				var_list_f2m_fuse.append(var)

		saver = tf.compat.v1.train.Saver(var_list=var_list_f2m)
		'''only for fusing aligned images'''
		# saver = tf.compat.v1.train.Saver(var_list=var_list_f2m_fuse)

		global_vars = tf.compat.v1.global_variables()
		sess.run(tf.compat.v1.variables_initializer(global_vars))
		_, _ = load(sess, saver, './models/finetuning/')

		for file in files:
			pic_num += 1
			names = file.split('/')[-1]
			name = names.split('.')[-2]
			format = names.split('.')[-1]
			print("\033[0;33;40m["+ str(pic_num) + "/" + str(len(files)) +"]: "+ name + ".png" + "\033[0m")

			rgb_img = scipy.misc.imread(test_path1 + file.split('/')[-1])
			nir_img = scipy.misc.imread(test_path2 + file.split('/')[-1])

			start_time = datetime.now()

			rgb_dimension = list(rgb_img.shape)
			nir_dimension = list(nir_img.shape)
			H = rgb_dimension[0] * 1.0
			W = rgb_dimension[1] * 1.0
			"resize"
			rgb_img_N = scipy.misc.imresize(rgb_img, size=(N, N))
			nir_img_N = scipy.misc.imresize(nir_img, size=(N, N))
			rgb_img_N = np.expand_dims(rgb_img_N, axis=0)
			nir_img_N = np.expand_dims(np.expand_dims(nir_img_N, axis=0), axis=-1)
			rgb_img_N = rgb_img_N.astype(np.float32)/255.0
			nir_img_N = nir_img_N.astype(np.float32) / 255.0

			FEED_DICT = {SOURCE_RGB: rgb_img_N, SOURCE_NIR: nir_img_N}
			fused_img = sess.run(f2m_model.fused_img, feed_dict=FEED_DICT)
			fused_img = scipy.misc.imresize(fused_img[0, :, :, :], (rgb_dimension[0], rgb_dimension[1])).astype(np.float32) / 255.0
			if not os.path.exists(save_path + 'fused_img/'):
				os.mkdir(save_path + 'fused_img/')
			scipy.misc.imsave(save_path + 'fused_img/' + name + '.' + format, fused_img)

			time = datetime.now() - start_time
			time=time.total_seconds()

			if pic_num>1:
				T.append(time)
				print("\nElapsed_time: %s" % (T[pic_num-2]))


		print("Time mean :%s, std: %s\n" % (np.mean(T), np.std(T)))

if __name__ == '__main__':
	main()


