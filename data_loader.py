# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
import copy
#==========================dataset load==========================
class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, image_label, tran, tran_label =sample['imidx'], sample['image'], sample['image_label'], sample['tran'], sample['tran_label']

		imidx = torch.from_numpy(imidx.copy())
		image = transforms.ToTensor()(image.copy())
		image_label = transforms.ToTensor()(image_label.copy())
		tran = transforms.ToTensor()(tran.copy())
		tran_label = transforms.ToTensor()(tran_label.copy())
		return {'imidx': imidx, 'image': image, 'image_label': image_label, 'tran': tran, "tran_label":tran_label}
class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, image_label, tran, tran_label =sample['imidx'], sample['image'], sample['image_label'], sample['tran'], sample['tran_label']

		if random.random() >= 0.5:
			image = image[::-1]
			image_label = image_label[::-1]

		h, w = image_label.shape[:2]
		new_h, new_w = self.output_size

		top = random.randrange(0, h - new_h,4)
		left = random.randrange(0, w - new_w,4)

		image = image[top: top + new_h, left: left + new_w]
		image_label = image_label[top: top + new_h, left: left + new_w]


		return {'imidx': imidx, 'image': image, 'image_label': image_label, 'tran': tran, "tran_label":tran_label}
class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,mask_name_list,tran_name_list,transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.mask_name_list = mask_name_list
		self.tran_name_list = tran_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		image = io.imread(self.image_name_list[idx])

		imidx = np.array([idx])

		image_label = io.imread(self.label_name_list[idx])

		# mask = io.imread(self.mask_name_list[idx])
		tran = np.max(image,axis=2) - np.min(image, axis=2)
		tran_label = tran


		# sample = {'imidx':imidx, 'image':image, 'label':label, 'mask':mask}
		sample = {'imidx': imidx, 'image': image, 'image_label': image_label, 'tran': tran, "tran_label":tran_label}

		if self.transform:
			sample = self.transform(sample)

		return sample
