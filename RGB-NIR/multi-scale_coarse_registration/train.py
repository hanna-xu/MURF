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
LEARNING_RATE = 0.0002

def train_affine_network(model, sess, trainset, merged1=None, writer=None, saver=None, save_path=None, task1_model_path=None, EPOCHES=20):
	start_time = datetime.now()
	num_imgs = trainset.shape[0]
	mod = num_imgs % model.batchsize
	n_batches = int(num_imgs // model.batchsize)

	print('Train images number %d, Batchsize: %d, Batches: %d.\n' % (num_imgs, model.batchsize, n_batches))
	if mod > 0:
		trainset = trainset[:-mod]

	var_list_des = tf.contrib.framework.get_trainable_variables(scope='des_extract')
	var_list_affine = tf.contrib.framework.get_trainable_variables(scope='affine_net')

	model.clip = [p.assign(tf.clip_by_value(p, -50, 50)) for p in var_list_affine]
	model.solver = tf.compat.v1.train.AdamOptimizer(learning_rate=model.lr, beta1=0.5, beta2=0.99).minimize(model.loss,
																						var_list=var_list_affine)
	initialize_uninitialized(sess)

	saver1 = tf.compat.v1.train.Saver(var_list = var_list_des)
	could_load, checkpoint_counter = load(sess=sess, saver=saver1, checkpoint_dir=task1_model_path)
	if could_load:
		print(" [*] Model of shared information extraction: load SUCCESS")
	else:
		print(" [!] Model of shared information extraction: load failed...")


	# # ** Start Training **
	for epoch in range(EPOCHES):
		state = np.random.get_state()
		np.random.shuffle(trainset)
		np.random.set_state(state)

		for batch in range(n_batches):
			model.step += 1
			rgb_batch = trainset[batch * model.batchsize:(batch * model.batchsize + model.batchsize), :, :, 0:3]
			nir_batch = trainset[batch * model.batchsize:(batch * model.batchsize + model.batchsize), :, :, 3]
			nir_batch = np.expand_dims(nir_batch, -1)

			if epoch < 5:
				lr = LEARNING_RATE
			else:
				lr = LEARNING_RATE * 0.998 ** ((model.step - 10 * n_batches) / 30)

			random_rotA = np.random.randint(4, size=model.batchsize)
			random_rotB = np.random.randint(4, size=model.batchsize)

			FEED_DICT = {model.RGB: rgb_batch, model.NIR: nir_batch, model.lr: lr}
			sess.run([model.solver, model.clip], feed_dict=FEED_DICT)
			print("Epoch: [%3d/%3d], Step: [%2d/%2d], lr: %s" % (epoch + 1, EPOCHES, model.step % n_batches, n_batches, lr))

			if model.step% 5 ==0:
				result = sess.run(merged1, feed_dict=FEED_DICT)
				writer.add_summary(result, model.step)
				writer.flush()

			is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)
			if is_last_step or model.step % logging_period == 0:
				elapsed_time = datetime.now() - start_time
				loss = sess.run(model.loss, feed_dict=FEED_DICT)
				print('Epoch: [%3d/%3d], Step: [%2d/%2d], Model step: %d, Elapsed_time: %s' % (
					epoch + 1, EPOCHES, model.step % n_batches, n_batches, model.step, elapsed_time))
				print('loss: %s, lr: %s\n ' % (loss, lr))

			if is_last_step or model.step % 100 == 0:
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				saver.save(sess, save_path + str(model.step) + '.ckpt')


