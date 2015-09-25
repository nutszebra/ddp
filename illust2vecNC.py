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

"""
  official page: http://illustration2vec.net/
  paper: http://illustration2vec.net/papers/illustration2vec-main.pdf
  caffe model: http://illustration2vec.net/models/illust2vec_ver200.caffemodel
  image_mean: http://illustration2vec.net/models/image_mean.npy
  layer structure is like this:
  [(u'conv1_1', [u'data'], [u'conv1_1']),
  (u'relu1_1', [u'conv1_1'], [u'conv1_1']),
  (u'pool1', [u'conv1_1'], [u'pool1']),
  (u'conv2_1', [u'pool1'], [u'conv2_1']),
  (u'relu2_1', [u'conv2_1'], [u'conv2_1']),
  (u'pool2', [u'conv2_1'], [u'pool2']),
  (u'conv3_1', [u'pool2'], [u'conv3_1']),
  (u'relu3_1', [u'conv3_1'], [u'conv3_1']),
  (u'conv3_2', [u'conv3_1'], [u'conv3_2']),
  (u'relu3_2', [u'conv3_2'], [u'conv3_2']),
  (u'pool3', [u'conv3_2'], [u'pool3']),
  (u'conv4_1', [u'pool3'], [u'conv4_1']),
  (u'relu4_1', [u'conv4_1'], [u'conv4_1']),
  (u'conv4_2', [u'conv4_1'], [u'conv4_2']),
  (u'relu4_2', [u'conv4_2'], [u'conv4_2']),
  (u'pool4', [u'conv4_2'], [u'pool4']),
  (u'conv5_1', [u'pool4'], [u'conv5_1']),
  (u'relu5_1', [u'conv5_1'], [u'conv5_1']),
  (u'conv5_2', [u'conv5_1'], [u'conv5_2']),
  (u'relu5_2', [u'conv5_2'], [u'conv5_2']),
  (u'pool5', [u'conv5_2'], [u'pool5']),
  (u'conv6_1', [u'pool5'], [u'conv6_1']),
  (u'relu6_1', [u'conv6_1'], [u'conv6_1']),
  (u'conv6_2', [u'conv6_1'], [u'conv6_2']),
  (u'relu6_2', [u'conv6_2'], [u'conv6_2']),
  (u'conv6_3', [u'conv6_2'], [u'conv6_3']),
  (u'relu6_3', [u'conv6_3'], [u'conv6_3']),
  (u'drop6_3', [u'conv6_3'], [u'conv6_3']),
  (u'encode1', [u'conv6_3'], [u'encode1']),
  (u'encode2', [u'encode1neuron'], [u'encode2'])]
"""

def getNeuralCode(directory, layer="conv5_1", gpu=-1):

  model = "illust2vec_ver200.caffemodel"
  #use illust2vec_ver200
  print('illust2vec_ver200 is being loaded!')
  #calculate load time
  timeMemory = time()
  func = caffe.CaffeFunction(model)
  print('illust2vec_ver200 was loaded!')
  print('It took ' + str(int(time() - timeMemory)) + " secondes")
  
  #gpu mode
  if gpu >= 0:
    cuda.init(gpu)
    func.to_gpu()
  
  in_size = 224
  # Constant mean over spatial pixels
  mean_image = np.load("illust2vec_image_mean.npy")
  print("neural code is extraced from layer " + layer)
  def neuralCode(x): #推測関数
    y, = func(inputs={'data': x}, outputs=[layer],
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
