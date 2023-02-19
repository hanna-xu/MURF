from __future__ import print_function
import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from datetime import datetime
from scipy.misc import imsave
import scipy.ndimage
import scipy.io as scio
from skimage import img_as_ubyte
import os
from utils import *

EPSILON = 1e-5
eps = 1e-8
logging_period = 10
patch_size = 256
LEARNING_RATE = 0.0005

def train_f2m(model, sess, trainset, merged1=None, writer=None, save_path=None, EPOCHES=20, name=None):
	start_time = datetime.now()
	num_imgs = trainset.shape[0]
	mod = num_imgs % model.batchsize
	n_batches = int(num_imgs // model.batchsize)
	print('Train images number %d, Batchsize: %d, Batches: %d.\n' % (num_imgs, model.batchsize, n_batches))
	if mod > 0:
		trainset = trainset[:-mod]

	var_list_f2m = tf.contrib.framework.get_trainable_variables(scope='f2m_net')
	var_list_f2m_offset = list()
	var_list_f2m_fuse = list()
	for i in var_list_f2m:
		if "offset" in i.name:
			var_list_f2m_offset.append(i)
		else:
			var_list_f2m_fuse.append(i)

	print('\n f2m var_list_fuse:')
	for v in var_list_f2m_fuse:
		print(v.name)

	print('\n f2m var_list_offset:')
	for v in var_list_f2m_offset:
		print(v.name)

	var_list_global = tf.compat.v1.global_variables()
	bn_moving_vars=[g for g in var_list_global if 'moving_mean' in g.name]
	bn_moving_vars+=[g for g in var_list_global if 'moving_variance' in g.name]

	print('\n f2m_batch_norm:')
	for v in bn_moving_vars:
		print(v.name)

	saver=tf.compat.v1.train.Saver(var_list = var_list_f2m_fuse+var_list_f2m_offset+bn_moving_vars,max_to_keep=5)
	
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		model.clip = [p.assign(tf.clip_by_value(p, -50, 50)) for p in var_list_f2m]
		model.fuse_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=model.lr, beta1=0.5, beta2=0.99).minimize(model.fuse_loss,
																							var_list=var_list_f2m_fuse)

		model.offset_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=model.lr, beta1=0.5, beta2=0.99).minimize(model.offset_loss,
																							var_list=var_list_f2m_offset)

	initialize_uninitialized(sess)
	could_load, checkpoint_counter = load(sess=sess, saver=tf.compat.v1.train.Saver(
		var_list=var_list_f2m_fuse + var_list_f2m_offset + bn_moving_vars), checkpoint_dir='./models/')
	model.step = checkpoint_counter


	# # ** Start Training **
	for epoch in range(EPOCHES):
		state = np.random.get_state()
		np.random.shuffle(trainset)
		np.random.set_state(state)

		for batch in range(n_batches):
			model.step += 1
			PET_batch = trainset[batch * model.batchsize:(batch * model.batchsize + model.batchsize), :, :, 0:3]
			MRI_batch = trainset[batch * model.batchsize:(batch * model.batchsize + model.batchsize), :, :, 3]
			MRI_batch = np.expand_dims(MRI_batch, -1)
			defor_field = create_defor_field(PET_batch, type='non_rigid') # rigid or non_rigid
			defor_re = defor_reverse(defor_field)
			# print("defor reverse shape: ", defor_re.shape)

			if epoch < 5:
				lr = LEARNING_RATE
			else:
				lr = LEARNING_RATE * 0.998 ** ((model.step - 5 * n_batches) / 30)

			FEED_DICT = {model.PET: PET_batch, model.MRI: MRI_batch, model.defor_field: defor_field, model.re_defor_field_gt: defor_re, model.lr: lr}
			if epoch < 50:
				sess.run([model.fuse_solver, model.clip], feed_dict=FEED_DICT)
				print("fuse")
			else:
				sess.run([model.offset_solver, model.clip], feed_dict=FEED_DICT)
				print("offset")


			if model.step % 5 ==0 and model.step > 150:
				print("\033[0;32;40m")
				print(name + '\n')
				print("\033[0m")
				print("Epoch: [%3d/%3d], Step: [%2d/%2d], lr: %s" % (epoch + 1, EPOCHES, model.step % n_batches, n_batches, lr))
				print("\033[0;33;40m")
				print("defor field max: ",
					np.max(np.sqrt(np.square(defor_field[:, :, :, 0]) + np.square(defor_field[:, :, :, 1]))) * np.max(
						[model.INPUT_H, model.INPUT_W]) / 2)
				print("\033[0m")
				result = sess.run(merged1, feed_dict=FEED_DICT)
				writer.add_summary(result, model.step)
				writer.flush()
    
			is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)

			if is_last_step or model.step % logging_period == 0:
				elapsed_time = datetime.now() - start_time
				# loss1, loss2 = sess.run([model.fuse_loss, model.offset_loss], feed_dict=FEED_DICT)
				print('Epoch: [%3d/%3d], Step: [%2d/%2d], Model step: %d, Elapsed_time: %s' % (
					epoch + 1, EPOCHES, model.step % n_batches, n_batches, model.step, elapsed_time))
				# print('fuse_loss: %s, offset_loss: %s, lr: %s\n ' % (loss1, loss2, lr))

			if is_last_step or model.step % 100 == 0:
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				saver.save(sess, save_path + str(model.step) + '.ckpt')