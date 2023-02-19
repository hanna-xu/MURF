from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
from scipy.io import loadmat
from train import train_f2m
from f2m_model import F2M_Model
from utils import *

data_path='aligned_train.h5'

patch_size = 256
EPOCHES = 50
NUM=2

def main():
	with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
		model = F2M_Model(BATCH_SIZE=8, INPUT_W=patch_size, INPUT_H=patch_size, is_training=True)
		SOURCE_CT = tf.placeholder(tf.float32, shape = (model.batchsize, patch_size, patch_size, 1), name = 'SOURCE1')
		SOURCE_MRI = tf.placeholder(tf.float32, shape = (model.batchsize, patch_size, patch_size, 1), name = 'SOURCE2')
		DEFOR_FIELD = tf.placeholder(tf.float32, shape=(model.batchsize, patch_size, patch_size, 2), name='defor_field')
		RE_DEFOR_FIELD_GT = tf.placeholder(tf.float32, shape=(model.batchsize, patch_size, patch_size, 2), name='re_defor_field_gt')
		model.f2m(SOURCE_CT, SOURCE_MRI, DEFOR_FIELD,RE_DEFOR_FIELD_GT, is_training=True)

		saver = tf.compat.v1.train.Saver(max_to_keep=5)
		gloss = tf.compat.v1.summary.scalar('fuse_loss', model.fuse_loss)
		loss1 = tf.compat.v1.summary.scalar('offset_loss', model.offset_loss)
		loss2 = tf.compat.v1.summary.scalar('smooth_loss', model.smooth_loss)
		SOURCE1 = tf.compat.v1.summary.image('CT', model.CT, max_outputs=NUM)
		SOURCE2 = tf.compat.v1.summary.image('MRI', model.MRI, max_outputs=NUM)
		dCT = tf.compat.v1.summary.image('dCT', model.dCT, max_outputs=NUM)
		fimg = tf.compat.v1.summary.image('fused_img', model.fused_img, max_outputs=NUM)
		dgt = tf.compat.v1.summary.image('defor_gt', model.dCT_defor_gt, max_outputs=NUM)
		merge_summary1 = tf.compat.v1.summary.merge(
			[SOURCE1, SOURCE2, gloss, dCT, fimg, loss1, loss2, dgt])

		print('Begin to train the network...\n')
		with tf.device('/cpu:0'):
			source_data = h5py.File(data_path,'r')
			source_data = source_data['data'][:]
			print("source_data shape:", source_data.shape)
			print("source_data max:", np.max(source_data))

		name = 'fuse_ct_mri'
		writer1 = tf.compat.v1.summary.FileWriter("./logs/" + name + "/", sess.graph)
		train_f2m(model=model, sess=sess, trainset=source_data, merged1=merge_summary1, writer=writer1, saver=saver,
						save_path='./models/', EPOCHES=EPOCHES, name=name)

if __name__ == '__main__':
	main()