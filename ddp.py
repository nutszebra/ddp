#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os, sys, re
import random

import cv2
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe

parser = argparse.ArgumentParser(
    description='detect duplicate pictures')
parser.add_argument('image',  help='Path to a folder that contains image file')
parser.add_argument('--gpu',  default=-1, help='The argument is number of gpu. -1 means cpu')
args = parser.parse_args()

#use alexnet
func = caffe.CaffeFunction("bvlc_alexnet.caffemodel")
print('Loaded', file=sys.stderr)
if args.gpu >= 0:
  cuda.init(args.gpu)
  func.to_gpu()

#input size
in_size = 227
mean_image = np.load("ilsvrc_2012_mean.npy")

# return neural code
def neuralCode(x):
  y, = func(inputs={'data': x}, outputs=['fc7'], train=False)
  return y.data[0]

cropwidth = 256 - in_size
start = cropwidth // 2
stop = start + in_size
mean_image = mean_image[:, start:stop, start:stop].copy()
target_shape = (256, 256)
output_side_length=256

#search pictures
picturePath = [picture for picture in os.listdir(args.image)
               if re.findall(r"\.png$|\.jpg$|\.JPG$|\.PNG$|\.JPEG$",picture)]
answer = {}

for picture in picturePath:
  #load image file
  image = cv2.imread(picture)
  #resize and crop
  height, width, depth = image.shape
  new_height = output_side_length
  new_width = output_side_length
  if height > width:
      new_height = output_side_length * height / width
  else:
      new_width = output_side_length * width / height
  resized_img = cv2.resize(image, (new_width, new_height))
  height_offset = (new_height - output_side_length) / 2
  width_offset = (new_width - output_side_length) / 2
  image= resized_img[height_offset:height_offset + output_side_length,
                     width_offset:width_offset + output_side_length]
  
  #subtract mean image
  image = image.transpose(2, 0, 1)
  image = image[:, start:stop, start:stop].astype(np.float32)
  image -= mean_image
  
  x_batch = np.ndarray(
      (1, 3, in_size,in_size), dtype=np.float32)
  x_batch[0]=image
  
  if args.gpu >= 0:
      x_batch=cuda.to_gpu(x_batch)
  
  #get neural code
  x = chainer.Variable(x_batch, volatile=True)
  answer[picture] = neuralCode(x)
