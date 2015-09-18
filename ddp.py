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
from chainer.functions import caffe

parser = argparse.ArgumentParser(
    description='detect duplicate pictures')
parser.add_argument('image', nargs="*",  help='Path to a folder that contains image file')
parser.add_argument('--gpu',  default=-1, help='The argument is number of gpu. -1 means cpu')
args = parser.parse_args()

print(args.image)
#use alexnet
print('alexnet is being loaded!')
timeMemory = time()
func = caffe.CaffeFunction("bvlc_alexnet.caffemodel")
print('alexnet was loaded!')
print('It took ' + str(int(time() - timeMemory)) + " secondes")
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

# calculate precision, recall, F
def calcResult(answer, threshold):
  tp = 0
  fp = 0
  tn = 0
  fn = 0
  item = {}
  item["fpFile"] = []
  item["tnFile"] = []
  #precision = tp / (tp + fp)
  #recall = tp / (tp + fn)
  for key in answer:
    duplicate = []
    for keykey in answer:
      if key == keykey:
        pass
      else:
        distance = answer[key].dot(answer[keykey])/(np.linalg.norm(answer[key])*np.linalg.nor(answer[keykey]))
        if distance >= threshold and key.split('/')[-1] == keykey.split('/')[-1]:
          tp = tp + 1
        elif distance >= threshold and key.split('/')[-1] != keykey.split('/')[-1]:
          tn = tn + 1
          item["tnFile"].append = {key: keykey}
        elif distance <= threshold and key.split('/')[-1] != keykey.split('/')[-1]:
          fn = fn + 1
        elif distance <= threshold and key.split('/')[-1] == keykey.split('/')[-1]:
          fp = fp + 1
          item["fpFile"].append = {key: keykey}
    item["tp"] = tp
    item["fp"] = tp
    item["tn"] = tp
    item["fn"] = tp
    item["recall"] = tp / (tp + fn)
    item["precision"] = tp / (tp + fp)
    return item
          

cropwidth = 256 - in_size
start = cropwidth // 2
stop = start + in_size
mean_image = mean_image[:, start:stop, start:stop].copy()
target_shape = (256, 256)
output_side_length=256

for folderPath in args.image:
  #search pictures
  picturePath = [picture for picture in os.listdir(folderPath)
                 if re.findall(r"\.png$|\.jpg$|\.JPG$|\.PNG$|\.JPEG$",picture)]
  print("you have totally " + str(len(picturePath)) + " pictures in " + folderPath)
  answer = {}
  count = 1
  
  for picture in picturePath:
    timeMemory = time()
    print('Number of pictures: ' + str(count))
    count = count + 1
    print('extracting neural code of ' + folderPath + "/" + picture)
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
    
    if args.gpu >= 0:
        x_batch=cuda.to_gpu(x_batch)
    
    #get neural code
    x = chainer.Variable(x_batch, volatile=True)
    answer[folderPath + "/" +  picture] = neuralCode(x)
    print('It took ' + str(int(10*(time() - timeMemory))/10.0) + " secondes")

result = calcResult(answer)
