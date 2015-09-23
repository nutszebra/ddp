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

def getNeuralCode(directory, layer="fc6", gpu = -1):

  model = "bvlc_alexnet.caffemodel"
  #use alexnet
  print('alexnet is being loaded!')
  #calculate load time
  timeMemory = time()
  func = caffe.CaffeFunction(model)
  print('alexnet was loaded!')
  print('It took ' + str(int(time() - timeMemory)) + " secondes")

  #gpu mode
  if gpu >= 0:
    cuda.init(gpu)
    func.to_gpu()

  in_size = 227
  mean_image = np.load("ilsvrc_2012_mean.npy")

  # return neural code
  print("neural code is extraced from layer " + layer)
  def neuralCode(x):
    y, = func(inputs={'data': x}, outputs=[layer], train=False)
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
  answer = {}
  count = 0
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
  newroke structure:
  [(u'conv1', [u'data'], [u'conv1']),
  (u'relu1', [u'conv1'], [u'conv1']),
  (u'norm1', [u'conv1'], [u'norm1']),
  (u'pool1', [u'norm1'], [u'pool1']),
  (u'conv2', [u'pool1'], [u'conv2']),
  (u'relu2', [u'conv2'], [u'conv2']),
  (u'norm2', [u'conv2'], [u'norm2']),
  (u'pool2', [u'norm2'], [u'pool2']),
  (u'conv3', [u'pool2'], [u'conv3']),
  (u'relu3', [u'conv3'], [u'conv3']),
  (u'conv4', [u'conv3'], [u'conv4']),
  (u'relu4', [u'conv4'], [u'conv4']),
  (u'conv5', [u'conv4'], [u'conv5']),
  (u'relu5', [u'conv5'], [u'conv5']),
  (u'pool5', [u'conv5'], [u'pool5']),
  (u'fc6', [u'pool5'], [u'fc6']),
  (u'relu6', [u'fc6'], [u'fc6']),
  (u'drop6', [u'fc6'], [u'fc6']),
  (u'fc7', [u'fc6'], [u'fc7']),
  (u'relu7', [u'fc7'], [u'fc7']),
  (u'drop7', [u'fc7'], [u'fc7']),
  (u'fc8', [u'fc7'], [u'fc8']),
  (u'loss', [u'fc8', u'label'], [])]
"""
