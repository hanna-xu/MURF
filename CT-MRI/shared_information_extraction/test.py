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
	test_path1 = './test_imgs/CT/'
	test_path2 = './test_imgs/MRI/'
	save_path = './des_results/'
	model_path ='./models/'

	files = listdir(test_path1)

	with tf.Graph().as_default(), tf.Session() as sess:
		model = Des_Extract_Model(BATCH_SIZE=1, INPUT_W=N, INPUT_H=N, is_training=False, equivariance=False)
		SOURCE_CT = tf.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE1')
		SOURCE_MRI = tf.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE2')
		model.des(SOURCE_CT, SOURCE_MRI)
		saver = tf.compat.v1.train.Saver(max_to_keep=5)

		var_list_CT = tf.contrib.framework.get_trainable_variables(scope='CT_Encoder')
		var_list_MRI = tf.contrib.framework.get_trainable_variables(scope='MRI_Encoder')
		var_list_descriptor = var_list_CT + var_list_MRI
		initialize_uninitialized(sess)
		_, _ = load(sess, saver, model_path)

		for file in files:
			name = file.split('/')[-1]
			print(name)
			CT_img = scipy.misc.imread(test_path1 + name)
			MRI_img = scipy.misc.imread(test_path2 + name)
			CT_dimension = list(CT_img.shape)
			MRI_dimension = list(MRI_img.shape)
			CT_img = scipy.misc.imresize(CT_img, (N, N))
			MRI_img = scipy.misc.imresize(MRI_img, size=(N, N))
			CT_img = np.expand_dims(np.expand_dims(CT_img, axis=0),axis=-1)
			MRI_img = np.expand_dims(np.expand_dims(MRI_img, axis=0), axis=-1)
			CT_img = CT_img.astype(np.float32) / 255.0
			MRI_img = MRI_img.astype(np.float32) / 255.0

			CT_des, MRI_des =  sess.run([model.CT_des, model.MRI_des], feed_dict={SOURCE_CT: CT_img, SOURCE_MRI: MRI_img})
			CT_des, MRI_des = normalize_common(CT_des, MRI_des)

			if not os.path.exists(save_path):
				os.mkdir(save_path)
			if not os.path.exists(save_path + '/CT/'):
				os.mkdir(save_path + '/CT/')
			if not os.path.exists(save_path + '/MRI/'):
				os.mkdir(save_path + '/MRI/')
			scipy.misc.imsave(save_path + 'CT/' + name, scipy.misc.imresize(CT_des[0, :, :, 0], (CT_dimension[0], CT_dimension[1])))
			scipy.misc.imsave(save_path + 'MRI/' + name, scipy.misc.imresize(MRI_des[0, :, :, 0], (MRI_dimension[0], MRI_dimension[1])))

if __name__ == '__main__':
	main()


