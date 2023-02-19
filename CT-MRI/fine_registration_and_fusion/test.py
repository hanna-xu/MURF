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

N = 256
def main():
	test_path1 = './test_imgs/warped_CT/'
	test_path2 = './test_imgs/MRI/'
	model_path ='./models/'
	save_path = './results/'
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	files = listdir(test_path1)

	pic_num = 0

	T=[]

	with tf.Graph().as_default(), tf.Session() as sess:
		f2m_model = F2M_Model(BATCH_SIZE=1, INPUT_W = None, INPUT_H = None, is_training=True)
		SOURCE_CT = tf.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE1')
		SOURCE_MRI = tf.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE2')
		f2m_model.f2m(SOURCE_CT, SOURCE_MRI, is_training=False)
		var_list_f2m = tf.contrib.framework.get_trainable_variables(scope='f2m_net')

		saver = tf.compat.v1.train.Saver(var_list=var_list_f2m)
		global_vars = tf.compat.v1.global_variables()
		sess.run(tf.compat.v1.variables_initializer(global_vars))
		_, _ = load(sess, saver, model_path)

		for file in files:
			pic_num += 1
			name = file.split('/')[-1]
			name = name.split('.')[-2]
			print("\033[0;33;40m["+ str(pic_num) + "/" + str(len(files)) +"]: "+ name + ".png" + "\033[0m")

			CT_img = scipy.misc.imread(test_path1 + file.split('/')[-1], flatten=True, mode='YCbCr')
			MRI_img = scipy.misc.imread(test_path2 + file.split('/')[-1])

			start_time = datetime.now()

			CT_dimension = list(CT_img.shape)
			MRI_dimension = list(MRI_img.shape)
			H = CT_dimension[0] * 1.0
			W = CT_dimension[1] * 1.0
			"resize"
			CT_img_N = scipy.misc.imresize(CT_img, size=(N, N))
			MRI_img_N = scipy.misc.imresize(MRI_img, size=(N, N))
			CT_img_N = np.expand_dims(np.expand_dims(CT_img_N, axis=0),axis=-1)
			MRI_img_N = np.expand_dims(np.expand_dims(MRI_img_N, axis=0), axis=-1)
			CT_img_N = CT_img_N.astype(np.float32)/255.0
			MRI_img_N = MRI_img_N.astype(np.float32) / 255.0
			print(CT_img_N.shape,MRI_img_N.shape)

			FEED_DICT = {SOURCE_CT: CT_img_N, SOURCE_MRI: MRI_img_N}
			fused_img = sess.run(f2m_model.fused_img, feed_dict=FEED_DICT)
			fused_img = scipy.misc.imresize(fused_img[0, :, :, 0], (CT_dimension[0], CT_dimension[1])).astype(np.float32) / 255.0
			if not os.path.exists(save_path + 'Fusion/'):
				os.mkdir(save_path + 'Fusion/')
			scipy.misc.imsave(save_path + 'Fusion/' + name + '.png', fused_img)

			time = datetime.now() - start_time
			time=time.total_seconds()

			if pic_num>1:
				T.append(time)
				print("\nElapsed_time: %s" % (T[pic_num-2]))
				print("Time mean :%s, std: %s\n" % (np.mean(T), np.std(T)))

if __name__ == '__main__':
	main()


