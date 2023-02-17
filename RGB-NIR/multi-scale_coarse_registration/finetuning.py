from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
from scipy.io import loadmat
from datetime import datetime
from utils import *
import scipy.misc
import cv2
import scipy.io as scio
from affine_model import Affine_Model, apply_affine_trans

patch_size = 1024
N = patch_size

batch_size=2
def main():
	test_path1 = './large_images_for_training/RGB/'
	test_path2 = './large_images_for_training/NIR/'
	save_path = './models_finetuning/'
	files = listdir(test_path1)
	step = 20000

	with tf.Graph().as_default(), tf.Session() as sess:
		model = Affine_Model(BATCH_SIZE=batch_size, INPUT_W=patch_size, INPUT_H=patch_size, is_training=True)
		SOURCE_RGB = tf.placeholder(tf.float32, shape=(batch_size, patch_size, patch_size, 3), name='SOURCE1')
		SOURCE_NIR = tf.placeholder(tf.float32, shape=(batch_size, patch_size, patch_size, 1), name='SOURCE2')
		model.affine(SOURCE_RGB, SOURCE_NIR, dropout=False)

		WSOURCE1, label, _ = apply_affine_trans(SOURCE_RGB, model.dtheta)
		WSOURCE1 = tf.multiply(WSOURCE1, tf.tile(label, [1, 1, 1, 3]))

		var_list_des = tf.contrib.framework.get_trainable_variables(scope='des_extract')
		var_list_affine = tf.contrib.framework.get_trainable_variables(scope='affine_net')
		var_list = var_list_affine + var_list_des

		model.clip = [p.assign(tf.clip_by_value(p, -50, 50)) for p in var_list_affine]
		model.solver = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.99).minimize(
			model.loss, var_list=var_list_affine)
		initialize_uninitialized(sess)

		saver1 = tf.compat.v1.train.Saver(var_list=var_list)
		global_vars = tf.compat.v1.global_variables()
		sess.run(tf.compat.v1.variables_initializer(global_vars))
		_, _ = load(sess, saver1, './models/')

		saver = tf.compat.v1.train.Saver(max_to_keep=5)
		NUM = 1
		loss1 = tf.compat.v1.summary.scalar('all_loss', model.loss)
		SOURCE1 = tf.compat.v1.summary.image('RGB', model.RGB, max_outputs=NUM)
		SOURCE2 = tf.compat.v1.summary.image('NIR', model.NIR, max_outputs=NUM)
		des1 = tf.compat.v1.summary.image('des_RGB', model.RGB_des, max_outputs=NUM)
		des2 = tf.compat.v1.summary.image('des_NIR', model.NIR_des, max_outputs=NUM)
		w_RGB = tf.compat.v1.summary.image('warped_RGB', model.warped_RGB, max_outputs=NUM)
		w_des1 = tf.compat.v1.summary.image('w_des_RGB', model.w_des1, max_outputs=NUM)
		w_des2 = tf.compat.v1.summary.image('w_des_NIR', model.w_des2, max_outputs=NUM)
		diff1 = tf.compat.v1.summary.image('diff_ori', model.diff_ori, max_outputs=NUM)
		diff2 = tf.compat.v1.summary.image('diff_after', model.diff_after, max_outputs=NUM)
		label = tf.compat.v1.summary.image('label', model.label, max_outputs=NUM)

		merge_summary = tf.compat.v1.summary.merge(
			[loss1, SOURCE1, SOURCE2, des1, des2, w_RGB, w_des1, w_des2, diff1, diff2, label])
		writer = tf.compat.v1.summary.FileWriter("./logs_finetuning/", sess.graph)

		start_time = datetime.now()
		for step in range(len(files) * 10):
			step += 1+20000
			for index in range(batch_size):
				x = np.random.randint(len(files))
				print("x", x)
				file = files[x]
				name = file.split('/')[-1]
				name = name.split('.')[-2]
				print("\033[0;37;40m"+ name + ".png" + "\033[0m")

				rgb_img = scipy.misc.imread(test_path1 + file.split('/')[-1])
				nir_img = scipy.misc.imread(test_path2 + file.split('/')[-1])
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

				if index==0:
					rgb_imgs_N = rgb_img_N
					nir_imgs_N = nir_img_N
				else:
					rgb_imgs_N = np.concatenate([rgb_imgs_N, rgb_img_N], axis=0)
					nir_imgs_N = np.concatenate([nir_imgs_N, nir_img_N], axis=0)

			FEED_DICT = {SOURCE_RGB: rgb_imgs_N, SOURCE_NIR: nir_imgs_N}
			sess.run([model.solver, model.clip], feed_dict=FEED_DICT)

			if step % 10==0:
				result = sess.run(merge_summary, feed_dict=FEED_DICT)
				writer.add_summary(result, step)
				writer.flush()

			elapsed_time = datetime.now() - start_time
			print("Step: [%2d/%2d], Elapsed_time: %s" % (step, len(files) * 10+20000, elapsed_time))

			if step % 50 == 0:
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				saver.save(sess, save_path + str(step) + '.ckpt')

if __name__ == '__main__':
	main()


