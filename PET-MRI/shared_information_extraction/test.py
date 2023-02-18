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

N=256

def main():
	test_path1 = './test_imgs/PET/'
	test_path2 = './test_imgs/MRI/'
	save_path = './des_results/'
	model_path = './models/'

	files = listdir(test_path1)

	with tf.Graph().as_default(), tf.Session() as sess:
		model = Des_Extract_Model(BATCH_SIZE=1, INPUT_W=N, INPUT_H=N, is_training=False, equivariance=False)
		SOURCE_PET = tf.placeholder(tf.float32, shape = (1, N, N, 3), name = 'SOURCE1')
		SOURCE_MRI = tf.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE2')
		model.des(SOURCE_PET, SOURCE_MRI)
		saver = tf.compat.v1.train.Saver(max_to_keep=5)

		var_list_PET = tf.contrib.framework.get_trainable_variables(scope='PET_Encoder')
		var_list_MRI = tf.contrib.framework.get_trainable_variables(scope='MRI_Encoder')
		var_list_descriptor = var_list_PET + var_list_MRI

		initialize_uninitialized(sess)
		_, _ = load(sess, saver, model_path)

		for file in files:
			name = file.split('/')[-1]
			print(name)
			PET_img = scipy.misc.imread(test_path1 + name)
			MRI_img = scipy.misc.imread(test_path2 + name)
			PET_dimension = list(PET_img.shape)
			MRI_dimension = list(MRI_img.shape)
			PET_img = scipy.misc.imresize(PET_img, (N, N))
			MRI_img = scipy.misc.imresize(MRI_img, size=(N, N))

			PET_img = np.expand_dims(PET_img, axis=0)
			MRI_img = np.expand_dims(np.expand_dims(MRI_img, axis=0), axis=-1)

			PET_img = PET_img.astype(np.float32) / 255.0
			MRI_img = MRI_img.astype(np.float32) / 255.0

			PET_des, MRI_des =  sess.run([model.PET_des, model.MRI_des], feed_dict={SOURCE_PET: PET_img, SOURCE_MRI: MRI_img})

			PET_des, MRI_des = normalize_common(PET_des, MRI_des)
			if not os.path.exists(save_path):
				os.mkdir(save_path)
			if not os.path.exists(save_path + '/PET/'):
				os.mkdir(save_path + '/PET/')
			if not os.path.exists(save_path + '/MRI/'):
				os.mkdir(save_path + '/MRI/')
			scipy.misc.imsave(save_path + 'PET/' + name, scipy.misc.imresize(PET_des[0, :, :, 0], (PET_dimension[0], PET_dimension[1])))
			scipy.misc.imsave(save_path + 'MRI/' + name, scipy.misc.imresize(MRI_des[0, :, :, 0], (MRI_dimension[0], MRI_dimension[1])))

if __name__ == '__main__':
	main()


