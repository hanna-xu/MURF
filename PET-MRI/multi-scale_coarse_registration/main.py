from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
from scipy.io import loadmat
from train import train_affine_network
from affine_model import Affine_Model

data_path = 'unaligned_train.h5'
task1_model_path = '../shared_information_extraction/models/'

patch_size = 256
EPOCHES = 100
NUM=5

def main():
	with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
		model = Affine_Model(BATCH_SIZE=32, INPUT_W=patch_size, INPUT_H=patch_size, is_training=True)
		SOURCE_PET = tf.placeholder(tf.float32, shape = (model.batchsize, patch_size, patch_size, 3), name = 'SOURCE1')
		SOURCE_MRI = tf.placeholder(tf.float32, shape = (model.batchsize, patch_size, patch_size, 1), name = 'SOURCE2')
		model.affine(SOURCE_PET, SOURCE_MRI)
		saver = tf.compat.v1.train.Saver(max_to_keep=5)
		gloss = tf.compat.v1.summary.scalar('loss', model.d1_loss)
		aloss = tf.compat.v1.summary.scalar('all_loss', model.loss)
		SOURCE1 = tf.compat.v1.summary.image('PET', model.PET, max_outputs=NUM)
		SOURCE2 = tf.compat.v1.summary.image('MRI', model.MRI, max_outputs=NUM)
		w_PET = tf.compat.v1.summary.image('warped_PET', model.warped_PET, max_outputs=NUM)
		w_des1 = tf.compat.v1.summary.image('w_des_PET', model.w_des1, max_outputs=NUM)
		w_des2 = tf.compat.v1.summary.image('w_des_MRI', model.w_des2, max_outputs=NUM)

		label = tf.compat.v1.summary.image('label', model.label, max_outputs=NUM)
		merge_summary1 = tf.compat.v1.summary.merge([SOURCE1, SOURCE2, gloss, w_PET, w_des1, w_des2, label, aloss])
		print('Begin to train the network...\n')
		with tf.device('/cpu:0'):
			source_data = h5py.File(data_path,'r')
			source_data = source_data['data'][:]
			print("source_data shape:", source_data.shape)

		name = 'multi-scale'
		writer1 = tf.compat.v1.summary.FileWriter("./logs/" + name + "/", sess.graph)
		train_affine_network(model=model, sess=sess, trainset=source_data, merged1=merge_summary1, writer=writer1, saver=saver,
						save_path='./models/', EPOCHES=EPOCHES,TASK1_MODEL_PATH=task1_model_path)

if __name__ == '__main__':
	main()