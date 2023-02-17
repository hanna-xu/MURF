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

data1_path = 'RGB_NIR_part1.h5'
data2_path = 'RGB_NIR_part2.h5'

patch_size = 256

EPOCHES = 50

NUM=3

def main():
	with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
		model = Des_Extract_Model(BATCH_SIZE=32, INPUT_W=patch_size, INPUT_H=patch_size, equivariance=True)
		RGB_input = tf.compat.v1.placeholder(tf.float32, shape=(model.batchsize, patch_size, patch_size, 3), name='RGB')
		NIR_input = tf.compat.v1.placeholder(tf.float32, shape=(model.batchsize, patch_size, patch_size, 1), name='NIR')
		model.des(RGB_input, NIR_input)

		saver = tf.compat.v1.train.Saver(max_to_keep=5)
		gloss = tf.compat.v1.summary.scalar('loss', model.loss)
		closs = tf.compat.v1.summary.scalar('contrast_loss', model.contrast_loss)
		SOURCE1 = tf.compat.v1.summary.image('RGB', model.RGB, max_outputs=NUM)
		SOURCE2 = tf.compat.v1.summary.image('NIR', model.NIR, max_outputs=NUM)
		des1 = tf.compat.v1.summary.image('des_RGB', model.RGB_des, max_outputs=NUM)
		des2 = tf.compat.v1.summary.image('des_NIR', model.NIR_des, max_outputs=NUM)
		similarities = tf.compat.v1.summary.image('similarities',
												  tf.expand_dims(tf.expand_dims(model.similarities, axis=0), axis=-1),
												  max_outputs=1)
		merge_summary1 = tf.compat.v1.summary.merge([gloss, closs, SOURCE1, SOURCE2, des1, des2, similarities])

		print('Begin to train the network...\n')
		with tf.device('/cpu:0'):
			source_data1 = h5py.File(data1_path, 'r')
			source_data1 = source_data1['data'][:]
			source_data1 = np.transpose(source_data1, (0, 3, 2, 1))
			source_data2 = h5py.File(data2_path, 'r')
			source_data2 = source_data2['data'][:]
			source_data2 = np.transpose(source_data2, (0, 3, 2, 1))
			source_data = source_data1 #np.concatenate([source_data1, source_data2], axis=0)
			print("source_data shape:", source_data.shape)

		name = 'des'
		writer1 = tf.compat.v1.summary.FileWriter("logs/" + name + "/", sess.graph)
		train_descriptor(model=model, sess=sess, trainset=source_data, merged1=merge_summary1, writer=writer1, saver=saver,
							save_path='./models/', EPOCHES=EPOCHES)

if __name__ == '__main__':
	main()
