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
from f2m_model import F2M_Model
from utils import *
import scipy.misc
import cv2
import scipy.io as scio

N = 1024
EPOCH = 2
batch_size = 1

def main():
	test_path1 = './large_images_for_training/RGB/'
	test_path2 = './large_images_for_training/NIR/'
	save_path = './models/finetuning/'
	files = listdir(test_path1)

	with tf.Graph().as_default(), tf.Session() as sess:
		f2m_model = F2M_Model(BATCH_SIZE=batch_size, INPUT_W=N, INPUT_H=N, is_training=True)
		SOURCE_RGB = tf.placeholder(tf.float32, shape=(batch_size, N, N, 3), name='SOURCE1')
		SOURCE_NIR = tf.placeholder(tf.float32, shape=(batch_size, N, N, 1), name='SOURCE2')
		DEFOR_FIELD = tf.placeholder(tf.float32, shape=(batch_size, N, N, 2), name='defor_field')
		RE_DEFOR_FIELD_GT = tf.placeholder(tf.float32, shape=(batch_size, N, N, 2), name='re_defor_field_gt')
		f2m_model.f2m(SOURCE_RGB, SOURCE_NIR, DEFOR_FIELD, RE_DEFOR_FIELD_GT, is_training=True)

		var_list_f2m = tf.contrib.framework.get_trainable_variables(scope='f2m_net')
		var_list_f2m_offset = list()
		var_list_f2m_fuse = list()
		for i in var_list_f2m:
			if "offset" in i.name:
				var_list_f2m_offset.append(i)
			else:
				var_list_f2m_fuse.append(i)

		f2m_model.offset_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5,
																   beta2=0.99).minimize(f2m_model.offset_loss,
																						var_list=var_list_f2m_offset)

		f2m_model.fuse_clip = [p.assign(tf.clip_by_value(p, -50, 50)) for p in var_list_f2m_fuse]
		f2m_model.offset_clip = [p.assign(tf.clip_by_value(p, -50, 50)) for p in var_list_f2m_offset]

		global_vars = tf.compat.v1.global_variables()
		sess.run(tf.compat.v1.variables_initializer(global_vars))
		_, begin_steps = load(sess, tf.compat.v1.train.Saver(var_list=var_list_f2m),
							  './models/f2m/')

		saver1 = tf.compat.v1.train.Saver(max_to_keep=5)
		NUM = 2
		offsetloss = tf.compat.v1.summary.scalar('smooth_loss', f2m_model.smooth_loss)
		defordiffloss = tf.compat.v1.summary.scalar('defor_diff', f2m_model.defor_diff)
		gloss = tf.compat.v1.summary.scalar('fuse_loss', f2m_model.fuse_loss)
		loss1 = tf.compat.v1.summary.scalar('sim_loss', f2m_model.sim_loss)
		loss2 = tf.compat.v1.summary.scalar('max_gradient_loss', f2m_model.max_gradient_loss)
		loss3= tf.compat.v1.summary.scalar('fimg_grad_loss', f2m_model.fimg_grad_loss)

		SOURCE1 = tf.compat.v1.summary.image('RGB', f2m_model.RGB, max_outputs=NUM)
		SOURCE2 = tf.compat.v1.summary.image('NIR', f2m_model.NIR, max_outputs=NUM)
		dRGB = tf.compat.v1.summary.image('dRGB', f2m_model.dRGB, max_outputs=NUM)
		dRGB_offset = tf.compat.v1.summary.image('dRGB_offset', f2m_model.a_offset, max_outputs=NUM)
		diff = tf.compat.v1.summary.image('diff_RGB_dRGB', tf.abs(f2m_model.dRGB - f2m_model.RGB), max_outputs=NUM)
		diff1 = tf.compat.v1.summary.image('RGB-NIR', f2m_model.mean_img_before, max_outputs=NUM)
		diff2 = tf.compat.v1.summary.image('dRGB-NIR', f2m_model.mean_img_after, max_outputs=NUM)
		fimg = tf.compat.v1.summary.image('fused_img', f2m_model.fused_img, max_outputs=NUM)
		dgt = tf.compat.v1.summary.image('defor_gt', f2m_model.drgb_defor_gt, max_outputs=NUM)

		merge_summary = tf.compat.v1.summary.merge(
			[SOURCE1, SOURCE2, offsetloss, dRGB, dRGB_offset, diff, diff1, diff2, fimg, dgt, gloss, loss1, loss2,
			 loss3, defordiffloss])
		writer = tf.compat.v1.summary.FileWriter('./logs/finetuning/', sess.graph)

		start_time = datetime.now()

		for step in range(len(files) * EPOCH):
			step += 1 + begin_steps
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
				rgb_img_N = rgb_img_N.astype(np.float32) / 255.0
				nir_img_N = nir_img_N.astype(np.float32) / 255.0

				if index == 0:
					rgb_imgs_N = rgb_img_N
					nir_imgs_N = nir_img_N
				else:
					rgb_imgs_N = np.concatenate([rgb_imgs_N, rgb_img_N], axis=0)
					nir_imgs_N = np.concatenate([nir_imgs_N, nir_img_N], axis=0)

			defor_field = create_defor_field(rgb_imgs_N, type='non_rigid')
			defor_re = defor_reverse(defor_field, half_window=10)
			print("defor reverse shape: ", defor_re.shape)
			FEED_DICT = {SOURCE_RGB: rgb_imgs_N, SOURCE_NIR: nir_imgs_N, DEFOR_FIELD: defor_field, RE_DEFOR_FIELD_GT: defor_re}
			sess.run([f2m_model.offset_solver, f2m_model.offset_clip], feed_dict=FEED_DICT)
			print("\033[0;33;40m")
			print("\noffset\n")
			print("\033[0m")
			if step % 2 == 0:
				result = sess.run(merge_summary, feed_dict=FEED_DICT)
				writer.add_summary(result, step)
				writer.flush()
				elapsed_time = datetime.now() - start_time
				print("Step: [%2d/%2d], Elapsed_time: %s" % (step, len(files) * EPOCH+begin_steps, elapsed_time))

			print("\033[0;33;40m")
			print("defor field max: ",
				  np.max(np.sqrt(np.square(defor_field[:, :, :, 0]) + np.square(defor_field[:, :, :, 1]))) * np.max(
					  [f2m_model.INPUT_H, f2m_model.INPUT_W]) / 2)
			print("\033[0m")

			if step % 50 == 0:
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				saver1.save(sess, save_path + str(step) + '.ckpt')


if __name__ == '__main__':
	main()
