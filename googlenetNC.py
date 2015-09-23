#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os, sys, re
import random
from time import time

import cv2
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import overwrite
from chainer.functions import caffe

def getNeuralCode(directory, layer="inception_4a/output", gpu=-1):

  model = "bvlc_googlenet.caffemodel"
  #use googlenet
  print('googlenet is being loaded!')
  #calculate load time
  timeMemory = time()
  func = caffe.CaffeFunction(model)
  print('googlenet was loaded!')
  print('It took ' + str(int(time() - timeMemory)) + " secondes")
  
  #gpu mode
  if gpu >= 0:
    cuda.init(gpu)
    func.to_gpu()
  
  in_size = 224
  # Constant mean over spatial pixels
  mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
  mean_image[0] = 104
  mean_image[1] = 117
  mean_image[2] = 123
  print("neural code is extraced from layer " + layer)
  def neuralCode(x): #推測関数
    y, = func(inputs={'data': x}, outputs=[layer],
              disable=['loss1/ave_pool', 'loss2/ave_pool'],
              train=False)
    return y.data[0]
  
  cropwidth = 256 - in_size
  start = cropwidth // 2
  stop = start + in_size
  mean_image = mean_image[:, start:stop, start:stop].copy()
  target_shape = (256, 256)
  output_side_length=256
  
  numPic = 0
  #count pictures
  for folderPath in directory:
    #search pictures
    picturePath = [picture for picture in os.listdir(folderPath)
                   if re.findall(r"\.png$|\.jpg$|\.JPG$|\.PNG$|\.JPEG$",picture)]
    print("you have " + str(len(picturePath)) + " pictures in " + folderPath)
    numPic = numPic + len(picturePath)
  
  print("you have totally " + str(numPic) + " pictures")
  count = 0
  answer = {}
  for folderPath in directory:
    #search pictures
    picturePath = [picture for picture in os.listdir(folderPath)
                   if re.findall(r"\.png$|\.jpg$|\.JPG$|\.PNG$|\.JPEG$",picture)]
  
    for picture in picturePath:
      timeMemory = time()
      count = count + 1
      #load image file
      image = cv2.imread(folderPath + "/" + picture)
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
  
      if gpu >= 0:
          x_batch=cuda.to_gpu(x_batch)
  
      #get neural code
      x = chainer.Variable(x_batch, volatile=True)
      answer[folderPath + "/" +  picture] = neuralCode(x)
      sen = overwrite.bar(count,numPic)
      overwrite.overwrite(sen)
  return answer

"""
network structure:

[(u'conv1/7x7_s2', [u'data'], [u'conv1/7x7_s2']),
(u'conv1/relu_7x7', [u'conv1/7x7_s2'], [u'conv1/7x7_s2']),
(u'pool1/3x3_s2', [u'conv1/7x7_s2'], [u'pool1/3x3_s2']),
(u'pool1/norm1', [u'pool1/3x3_s2'], [u'pool1/norm1']),
(u'conv2/3x3_reduce', [u'pool1/norm1'], [u'conv2/3x3_reduce']),
(u'conv2/relu_3x3_reduce', [u'conv2/3x3_reduce'], [u'conv2/3x3_reduce']),
(u'conv2/3x3', [u'conv2/3x3_reduce'], [u'conv2/3x3']),
(u'conv2/relu_3x3', [u'conv2/3x3'], [u'conv2/3x3']),
(u'conv2/norm2', [u'conv2/3x3'], [u'conv2/norm2']),
(u'pool2/3x3_s2', [u'conv2/norm2'], [u'pool2/3x3_s2']),
(u'inception_3a/1x1', [u'pool2/3x3_s2'], [u'inception_3a/1x1']),
(u'inception_3a/relu_1x1', [u'inception_3a/1x1'], [u'inception_3a/1x1']),
(u'inception_3a/3x3_reduce', [u'pool2/3x3_s2'], [u'inception_3a/3x3_reduce']),
(u'inception_3a/relu_3x3_reduce',
[u'inception_3a/3x3_reduce'],
[u'inception_3a/3x3_reduce']),
(u'inception_3a/3x3', [u'inception_3a/3x3_reduce'], [u'inception_3a/3x3']),
(u'inception_3a/relu_3x3', [u'inception_3a/3x3'], [u'inception_3a/3x3']),
(u'inception_3a/5x5_reduce', [u'pool2/3x3_s2'], [u'inception_3a/5x5_reduce']),
(u'inception_3a/relu_5x5_reduce',
[u'inception_3a/5x5_reduce'],
[u'inception_3a/5x5_reduce']),
(u'inception_3a/5x5', [u'inception_3a/5x5_reduce'], [u'inception_3a/5x5']),
(u'inception_3a/relu_5x5', [u'inception_3a/5x5'], [u'inception_3a/5x5']),
(u'inception_3a/pool', [u'pool2/3x3_s2'], [u'inception_3a/pool']),
(u'inception_3a/pool_proj',
[u'inception_3a/pool'],
[u'inception_3a/pool_proj']),
(u'inception_3a/relu_pool_proj',
[u'inception_3a/pool_proj'],
[u'inception_3a/pool_proj']),
(u'inception_3a/output',
[u'inception_3a/1x1',
u'inception_3a/3x3',
u'inception_3a/5x5',
u'inception_3a/pool_proj'],
[u'inception_3a/output']),
(u'inception_3b/1x1', [u'inception_3a/output'], [u'inception_3b/1x1']),
(u'inception_3b/relu_1x1', [u'inception_3b/1x1'], [u'inception_3b/1x1']),
(u'inception_3b/3x3_reduce',
[u'inception_3a/output'],
[u'inception_3b/3x3_reduce']),
(u'inception_3b/relu_3x3_reduce',
[u'inception_3b/3x3_reduce'],
[u'inception_3b/3x3_reduce']),
(u'inception_3b/3x3', [u'inception_3b/3x3_reduce'], [u'inception_3b/3x3']),
(u'inception_3b/relu_3x3', [u'inception_3b/3x3'], [u'inception_3b/3x3']),
(u'inception_3b/5x5_reduce',
[u'inception_3a/output'],
[u'inception_3b/5x5_reduce']),
(u'inception_3b/relu_5x5_reduce',
[u'inception_3b/5x5_reduce'],
[u'inception_3b/5x5_reduce']),
(u'inception_3b/5x5', [u'inception_3b/5x5_reduce'], [u'inception_3b/5x5']),
(u'inception_3b/relu_5x5', [u'inception_3b/5x5'], [u'inception_3b/5x5']),
(u'inception_3b/pool', [u'inception_3a/output'], [u'inception_3b/pool']),
(u'inception_3b/pool_proj',
[u'inception_3b/pool'],
[u'inception_3b/pool_proj']),
(u'inception_3b/relu_pool_proj',
[u'inception_3b/pool_proj'],
[u'inception_3b/pool_proj']),
(u'inception_3b/output',
[u'inception_3b/1x1',
u'inception_3b/3x3',
u'inception_3b/5x5',
u'inception_3b/pool_proj'],
[u'inception_3b/output']),
(u'pool3/3x3_s2', [u'inception_3b/output'], [u'pool3/3x3_s2']),
(u'inception_4a/1x1', [u'pool3/3x3_s2'], [u'inception_4a/1x1']),
(u'inception_4a/relu_1x1', [u'inception_4a/1x1'], [u'inception_4a/1x1']),
(u'inception_4a/3x3_reduce', [u'pool3/3x3_s2'], [u'inception_4a/3x3_reduce']),
(u'inception_4a/relu_3x3_reduce',
[u'inception_4a/3x3_reduce'],
[u'inception_4a/3x3_reduce']),
(u'inception_4a/3x3', [u'inception_4a/3x3_reduce'], [u'inception_4a/3x3']),
(u'inception_4a/relu_3x3', [u'inception_4a/3x3'], [u'inception_4a/3x3']),
(u'inception_4a/5x5_reduce', [u'pool3/3x3_s2'], [u'inception_4a/5x5_reduce']),
(u'inception_4a/relu_5x5_reduce',
[u'inception_4a/5x5_reduce'],
[u'inception_4a/5x5_reduce']),
(u'inception_4a/5x5', [u'inception_4a/5x5_reduce'], [u'inception_4a/5x5']),
(u'inception_4a/relu_5x5', [u'inception_4a/5x5'], [u'inception_4a/5x5']),
(u'inception_4a/pool', [u'pool3/3x3_s2'], [u'inception_4a/pool']),
(u'inception_4a/pool_proj',
[u'inception_4a/pool'],
[u'inception_4a/pool_proj']),
(u'inception_4a/relu_pool_proj',
[u'inception_4a/pool_proj'],
[u'inception_4a/pool_proj']),
(u'inception_4a/output',
[u'inception_4a/1x1',
u'inception_4a/3x3',
u'inception_4a/5x5',
u'inception_4a/pool_proj'],
[u'inception_4a/output']),
(u'loss1/ave_pool', [u'inception_4a/output'], [u'loss1/ave_pool']),
(u'loss1/conv', [u'loss1/ave_pool'], [u'loss1/conv']),
(u'loss1/relu_conv', [u'loss1/conv'], [u'loss1/conv']),
(u'loss1/fc', [u'loss1/conv'], [u'loss1/fc']),
(u'loss1/relu_fc', [u'loss1/fc'], [u'loss1/fc']),
(u'loss1/drop_fc', [u'loss1/fc'], [u'loss1/fc']),
(u'loss1/classifier', [u'loss1/fc'], [u'loss1/classifier']),
(u'loss1/loss', [u'loss1/classifier', u'label'], [u'loss1/loss1']),
(u'inception_4b/1x1', [u'inception_4a/output'], [u'inception_4b/1x1']),
(u'inception_4b/relu_1x1', [u'inception_4b/1x1'], [u'inception_4b/1x1']),
(u'inception_4b/3x3_reduce',
[u'inception_4a/output'],
[u'inception_4b/3x3_reduce']),
(u'inception_4b/relu_3x3_reduce',
[u'inception_4b/3x3_reduce'],
[u'inception_4b/3x3_reduce']),
(u'inception_4b/3x3', [u'inception_4b/3x3_reduce'], [u'inception_4b/3x3']),
(u'inception_4b/relu_3x3', [u'inception_4b/3x3'], [u'inception_4b/3x3']),
(u'inception_4b/5x5_reduce',
[u'inception_4a/output'],
[u'inception_4b/5x5_reduce']),
(u'inception_4b/relu_5x5_reduce',
[u'inception_4b/5x5_reduce'],
[u'inception_4b/5x5_reduce']),
(u'inception_4b/5x5', [u'inception_4b/5x5_reduce'], [u'inception_4b/5x5']),
(u'inception_4b/relu_5x5', [u'inception_4b/5x5'], [u'inception_4b/5x5']),
(u'inception_4b/pool', [u'inception_4a/output'], [u'inception_4b/pool']),
(u'inception_4b/pool_proj',
[u'inception_4b/pool'],
[u'inception_4b/pool_proj']),
(u'inception_4b/relu_pool_proj',
[u'inception_4b/pool_proj'],
[u'inception_4b/pool_proj']),
(u'inception_4b/output',
[u'inception_4b/1x1',
u'inception_4b/3x3',
u'inception_4b/5x5',
u'inception_4b/pool_proj'],
[u'inception_4b/output']),
(u'inception_4c/1x1', [u'inception_4b/output'], [u'inception_4c/1x1']),
(u'inception_4c/relu_1x1', [u'inception_4c/1x1'], [u'inception_4c/1x1']),
(u'inception_4c/3x3_reduce',
[u'inception_4b/output'],
[u'inception_4c/3x3_reduce']),
(u'inception_4c/relu_3x3_reduce',
[u'inception_4c/3x3_reduce'],
[u'inception_4c/3x3_reduce']),
(u'inception_4c/3x3', [u'inception_4c/3x3_reduce'], [u'inception_4c/3x3']),
(u'inception_4c/relu_3x3', [u'inception_4c/3x3'], [u'inception_4c/3x3']),
(u'inception_4c/5x5_reduce',
[u'inception_4b/output'],
[u'inception_4c/5x5_reduce']),
(u'inception_4c/relu_5x5_reduce',
[u'inception_4c/5x5_reduce'],
[u'inception_4c/5x5_reduce']),
(u'inception_4c/5x5', [u'inception_4c/5x5_reduce'], [u'inception_4c/5x5']),
(u'inception_4c/relu_5x5', [u'inception_4c/5x5'], [u'inception_4c/5x5']),
(u'inception_4c/pool', [u'inception_4b/output'], [u'inception_4c/pool']),
(u'inception_4c/pool_proj',
[u'inception_4c/pool'],
[u'inception_4c/pool_proj']),
(u'inception_4c/relu_pool_proj',
[u'inception_4c/pool_proj'],
[u'inception_4c/pool_proj']),
(u'inception_4c/output',
[u'inception_4c/1x1',
u'inception_4c/3x3',
u'inception_4c/5x5',
u'inception_4c/pool_proj'],
[u'inception_4c/output']),
(u'inception_4d/1x1', [u'inception_4c/output'], [u'inception_4d/1x1']),
(u'inception_4d/relu_1x1', [u'inception_4d/1x1'], [u'inception_4d/1x1']),
(u'inception_4d/3x3_reduce',
[u'inception_4c/output'],
[u'inception_4d/3x3_reduce']),
(u'inception_4d/relu_3x3_reduce',
[u'inception_4d/3x3_reduce'],
[u'inception_4d/3x3_reduce']),
(u'inception_4d/3x3', [u'inception_4d/3x3_reduce'], [u'inception_4d/3x3']),
(u'inception_4d/relu_3x3', [u'inception_4d/3x3'], [u'inception_4d/3x3']),
(u'inception_4d/5x5_reduce',
[u'inception_4c/output'],
[u'inception_4d/5x5_reduce']),
(u'inception_4d/relu_5x5_reduce',
[u'inception_4d/5x5_reduce'],
[u'inception_4d/5x5_reduce']),
(u'inception_4d/5x5', [u'inception_4d/5x5_reduce'], [u'inception_4d/5x5']),
(u'inception_4d/relu_5x5', [u'inception_4d/5x5'], [u'inception_4d/5x5']),
(u'inception_4d/pool', [u'inception_4c/output'], [u'inception_4d/pool']),
(u'inception_4d/pool_proj',
[u'inception_4d/pool'],
[u'inception_4d/pool_proj']),
(u'inception_4d/relu_pool_proj',
[u'inception_4d/pool_proj'],
[u'inception_4d/pool_proj']),
(u'inception_4d/output',
[u'inception_4d/1x1',
u'inception_4d/3x3',
u'inception_4d/5x5',
u'inception_4d/pool_proj'],
[u'inception_4d/output']),
(u'loss2/ave_pool', [u'inception_4d/output'], [u'loss2/ave_pool']),
(u'loss2/conv', [u'loss2/ave_pool'], [u'loss2/conv']),
(u'loss2/relu_conv', [u'loss2/conv'], [u'loss2/conv']),
(u'loss2/fc', [u'loss2/conv'], [u'loss2/fc']),
(u'loss2/relu_fc', [u'loss2/fc'], [u'loss2/fc']),
(u'loss2/drop_fc', [u'loss2/fc'], [u'loss2/fc']),
(u'loss2/classifier', [u'loss2/fc'], [u'loss2/classifier']),
(u'loss2/loss', [u'loss2/classifier', u'label'], [u'loss2/loss1']),
(u'inception_4e/1x1', [u'inception_4d/output'], [u'inception_4e/1x1']),
(u'inception_4e/relu_1x1', [u'inception_4e/1x1'], [u'inception_4e/1x1']),
(u'inception_4e/3x3_reduce',
[u'inception_4d/output'],
[u'inception_4e/3x3_reduce']),
(u'inception_4e/relu_3x3_reduce',
[u'inception_4e/3x3_reduce'],
[u'inception_4e/3x3_reduce']),
(u'inception_4e/3x3', [u'inception_4e/3x3_reduce'], [u'inception_4e/3x3']),
(u'inception_4e/relu_3x3', [u'inception_4e/3x3'], [u'inception_4e/3x3']),
(u'inception_4e/5x5_reduce',
[u'inception_4d/output'],
[u'inception_4e/5x5_reduce']),
(u'inception_4e/relu_5x5_reduce',
[u'inception_4e/5x5_reduce'],
[u'inception_4e/5x5_reduce']),
(u'inception_4e/5x5', [u'inception_4e/5x5_reduce'], [u'inception_4e/5x5']),
(u'inception_4e/relu_5x5', [u'inception_4e/5x5'], [u'inception_4e/5x5']),
(u'inception_4e/pool', [u'inception_4d/output'], [u'inception_4e/pool']),
(u'inception_4e/pool_proj',
[u'inception_4e/pool'],
[u'inception_4e/pool_proj']),
(u'inception_4e/relu_pool_proj',
[u'inception_4e/pool_proj'],
[u'inception_4e/pool_proj']),
(u'inception_4e/output',
[u'inception_4e/1x1',
u'inception_4e/3x3',
u'inception_4e/5x5',
u'inception_4e/pool_proj'],
[u'inception_4e/output']),
(u'pool4/3x3_s2', [u'inception_4e/output'], [u'pool4/3x3_s2']),
(u'inception_5a/1x1', [u'pool4/3x3_s2'], [u'inception_5a/1x1']),
(u'inception_5a/relu_1x1', [u'inception_5a/1x1'], [u'inception_5a/1x1']),
(u'inception_5a/3x3_reduce', [u'pool4/3x3_s2'], [u'inception_5a/3x3_reduce']),
(u'inception_5a/relu_3x3_reduce',
[u'inception_5a/3x3_reduce'],
[u'inception_5a/3x3_reduce']),
(u'inception_5a/3x3', [u'inception_5a/3x3_reduce'], [u'inception_5a/3x3']),
(u'inception_5a/relu_3x3', [u'inception_5a/3x3'], [u'inception_5a/3x3']),
(u'inception_5a/5x5_reduce', [u'pool4/3x3_s2'], [u'inception_5a/5x5_reduce']),
(u'inception_5a/relu_5x5_reduce',
[u'inception_5a/5x5_reduce'],
[u'inception_5a/5x5_reduce']),
(u'inception_5a/5x5', [u'inception_5a/5x5_reduce'], [u'inception_5a/5x5']),
(u'inception_5a/relu_5x5', [u'inception_5a/5x5'], [u'inception_5a/5x5']),
(u'inception_5a/pool', [u'pool4/3x3_s2'], [u'inception_5a/pool']),
(u'inception_5a/pool_proj',
[u'inception_5a/pool'],
[u'inception_5a/pool_proj']),
(u'inception_5a/relu_pool_proj',
[u'inception_5a/pool_proj'],
[u'inception_5a/pool_proj']),
(u'inception_5a/output',
[u'inception_5a/1x1',
u'inception_5a/3x3',
u'inception_5a/5x5',
u'inception_5a/pool_proj'],
[u'inception_5a/output']),
(u'inception_5b/1x1', [u'inception_5a/output'], [u'inception_5b/1x1']),
(u'inception_5b/relu_1x1', [u'inception_5b/1x1'], [u'inception_5b/1x1']),
(u'inception_5b/3x3_reduce',
[u'inception_5a/output'],
[u'inception_5b/3x3_reduce']),
(u'inception_5b/relu_3x3_reduce',
[u'inception_5b/3x3_reduce'],
[u'inception_5b/3x3_reduce']),
(u'inception_5b/3x3', [u'inception_5b/3x3_reduce'], [u'inception_5b/3x3']),
(u'inception_5b/relu_3x3', [u'inception_5b/3x3'], [u'inception_5b/3x3']),
(u'inception_5b/5x5_reduce',
[u'inception_5a/output'],
[u'inception_5b/5x5_reduce']),
(u'inception_5b/relu_5x5_reduce',
[u'inception_5b/5x5_reduce'],
[u'inception_5b/5x5_reduce']),
(u'inception_5b/5x5', [u'inception_5b/5x5_reduce'], [u'inception_5b/5x5']),
(u'inception_5b/relu_5x5', [u'inception_5b/5x5'], [u'inception_5b/5x5']),
(u'inception_5b/pool', [u'inception_5a/output'], [u'inception_5b/pool']),
(u'inception_5b/pool_proj',
[u'inception_5b/pool'],
[u'inception_5b/pool_proj']),
(u'inception_5b/relu_pool_proj',
[u'inception_5b/pool_proj'],
[u'inception_5b/pool_proj']),
(u'inception_5b/output',
[u'inception_5b/1x1',
u'inception_5b/3x3',
u'inception_5b/5x5',
u'inception_5b/pool_proj'],
[u'inception_5b/output']),
(u'pool5/7x7_s1', [u'inception_5b/output'], [u'pool5/7x7_s1']),
(u'pool5/drop_7x7_s1', [u'pool5/7x7_s1'], [u'pool5/7x7_s1']),
(u'loss3/classifier', [u'pool5/7x7_s1'], [u'loss3/classifier']),
(u'loss3/loss3', [u'loss3/classifier', u'label'], [u'loss3/loss3'])]

"""
