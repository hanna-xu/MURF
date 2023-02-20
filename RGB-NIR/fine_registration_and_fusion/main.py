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

data1_path = 'aligned_RGB_NIR_part1.h5'
data2_path = 'aligned_RGB_NIR_part2.h5'


patch_size = 256
EPOCHES = 15
NUM=2

def main():
	with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
		model = F2M_Model(BATCH_SIZE=4, INPUT_W=patch_size, INPUT_H=patch_size, is_training=True)

		SOURCE_RGB = tf.placeholder(tf.float32, shape = (model.batchsize, patch_size, patch_size, 3), name = 'SOURCE1')
		SOURCE_NIR = tf.placeholder(tf.float32, shape = (model.batchsize, patch_size, patch_size, 1), name = 'SOURCE2')
		DEFOR_FIELD = tf.placeholder(tf.float32, shape=(model.batchsize, patch_size, patch_size, 2), name='defor_field')
		RE_DEFOR_FIELD_GT = tf.placeholder(tf.float32, shape=(model.batchsize, patch_size, patch_size, 2), name='re_defor_field_gt')
		model.f2m(SOURCE_RGB, SOURCE_NIR, DEFOR_FIELD, RE_DEFOR_FIELD_GT, is_training=True)

		saver = tf.compat.v1.train.Saver(max_to_keep=5)
		gloss = tf.compat.v1.summary.scalar('fuse_loss', model.fuse_loss)
		loss1 = tf.compat.v1.summary.scalar('sim_loss', model.sim_loss)
		loss2 = tf.compat.v1.summary.scalar('max_gradient_loss', model.max_gradient_loss)
		loss3 = tf.compat.v1.summary.scalar('offset_loss', model.offset_loss)
		loss4 = tf.compat.v1.summary.scalar('smooth_loss', model.smooth_loss)
		ofs = tf.compat.v1.summary.scalar('defor_diff', model.defor_diff)

		SOURCE1 = tf.compat.v1.summary.image('RGB', model.RGB, max_outputs=NUM)
		SOURCE2 = tf.compat.v1.summary.image('NIR', model.NIR, max_outputs=NUM)
		dRGB = tf.compat.v1.summary.image('dRGB', model.dRGB, max_outputs=NUM)
		diff = tf.compat.v1.summary.image('diff_RGB_dRGB', tf.abs(model.dRGB-model.RGB), max_outputs=NUM)
		diff1 = tf.compat.v1.summary.image('RGB-NIR', model.mean_img_before, max_outputs=NUM)
		diff2 = tf.compat.v1.summary.image('dRGB-NIR', model.mean_img_after, max_outputs=NUM)
		d1 = tf.compat.v1.summary.image('drgb_offset', model.a_offset, max_outputs=NUM)
		fimg = tf.compat.v1.summary.image('fused_img', model.fused_img, max_outputs=NUM)
		dgt = tf.compat.v1.summary.image('defor_gt', model.drgb_defor_gt, max_outputs=NUM)

		merge_summary1 = tf.compat.v1.summary.merge(
			[SOURCE1, SOURCE2, gloss, dRGB, diff, diff1, d1, diff2, fimg, loss1, loss2, loss3, loss4, ofs, dgt])


		print('Begin to train the network...\n')
		with tf.device('/cpu:0'):
			source_data1 = h5py.File(data1_path, 'r')
			source_data1 = source_data1['data'][:]
			source_data1 = np.transpose(source_data1, (0, 3, 2, 1))
			source_data2 = h5py.File(data2_path, 'r')
			source_data2 = source_data2['data'][:]
			source_data2 = np.transpose(source_data2, (0, 3, 2, 1))
			source_data = np.concatenate([source_data1, source_data2], axis=0)
			print("source_data shape:", source_data.shape)

		name = 'f2m'
		writer1 = tf.compat.v1.summary.FileWriter("./logs/" + name + "/", sess.graph)
		train_f2m(model=model, sess=sess, trainset=source_data, merged1=merge_summary1, writer=writer1, saver=saver,
						save_path='./models/f2m/', EPOCHES=EPOCHES, name=name)

if __name__ == '__main__':
	main()