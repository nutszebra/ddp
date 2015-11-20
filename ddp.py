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
import alexnetNC
import googlenetNC
import illust2vecNC
import vlad

parser = argparse.ArgumentParser(
    description='detect duplicate pictures')
parser.add_argument('--model', default = "googlenet", help='select model')
parser.add_argument('--gpu',  default=-1, help='The argument is number of gpu. -1 means cpu')
parser.add_argument('--layer', default="default", help='layer to generate neural code')
parser.add_argument('--number', default=100, help='the number to execute k-means')
parser.add_argument('image', nargs="*",  help='Path to a folder that contains image file')
args = parser.parse_args()

# calculate precision, recall
def calcResult(answer, threshold):
  item = {}
  item["allFiles"] = {}
  item["fpFile"] = []
  item["fnFile"] = []
  #precision = tp / (tp + fp)
  #recall = tp / (tp + fn)
  for key in answer:
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for keykey in answer:
      if key == keykey:
        pass
      else:
        distance = answer[key].dot(answer[keykey])/(np.linalg.norm(answer[key])*np.linalg.norm(answer[keykey]))
        if distance >= threshold and key.split('/')[-1] == keykey.split('/')[-1]:
          tp = tp + 1
        elif distance >= threshold and key.split('/')[-1] != keykey.split('/')[-1]:
          fp = fp + 1
          item["fpFile"].append({key: keykey})
        elif distance <= threshold and key.split('/')[-1] != keykey.split('/')[-1]:
          tn = tn + 1
        elif distance <= threshold and key.split('/')[-1] == keykey.split('/')[-1]:
          fn = fn + 1
          item["fnFile"].append({key: keykey})
    item["allFiles"][key]={}
    item["allFiles"][key]["tp"] = tp
    item["allFiles"][key]["fp"] = fp
    item["allFiles"][key]["tn"] = tn
    item["allFiles"][key]["fn"] = fn

    if (tp + fn) != 0:
      item["allFiles"][key]["recall-positive"] = float(tp) / float(tp + fn)
    else:
      item["allFiles"][key]["recall-positive"] = "Inf"

    if (tp + fp) != 0:
      item["allFiles"][key]["precision-positive"] = float(tp) / float(tp + fp)
    else:
      item["allFiles"][key]["precision-positive"] = "Inf"

    if (tn + fp) != 0:
      item["allFiles"][key]["recall-negative"] = float(tn) / float(tn + fp)
    else:
      item["allFiles"][key]["recall-negative"] = "Inf"

    if (tn + fn) != 0:
      item["allFiles"][key]["precision-negative"] = float(tn) / float(tn + fn)
    else:
      item["allFiles"][key]["precision-negative"] = "Inf"

  mAPP = 0
  mARP = 0
  mAPN = 0
  mARN = 0

  numPrePInf = 0
  numRecPInf = 0
  numPreNInf = 0
  numRecNInf = 0

  for key in item["allFiles"]:
    if item["allFiles"][key]["precision-positive"] != "Inf":
      mAPP = mAPP + item["allFiles"][key]["precision-positive"]
    else:
      numPrePInf = numPrePInf + 1

    if item["allFiles"][key]["recall-positive"] != "Inf":
      mARP = mARP + item["allFiles"][key]["recall-positive"]
    else:
      numRecPInf = numRecPInf + 1

    if item["allFiles"][key]["precision-negative"] != "Inf":
      mAPN = mAPN + item["allFiles"][key]["precision-negative"]
    else:
      numPreNInf = numPreNInf + 1

    if item["allFiles"][key]["recall-negative"] != "Inf":
      mARN = mARN + item["allFiles"][key]["recall-negative"]
    else:
      numRecNInf = numRecNInf + 1


  item["mAPP"] = float(mAPP) / float(len(item["allFiles"]))
  item["mARP"] = float(mARP) / float(len(item["allFiles"]))
  item["mAPN"] = float(mAPN) / float(len(item["allFiles"]))
  item["mARN"] = float(mARN) / float(len(item["allFiles"]))

  item["numPrePInf"] = numPrePInf
  item["numRecPInf"] = numRecPInf
  item["numPreNInf"] = numPreNInf
  item["numRecNInf"] = numRecNInf

  return item

def showPrecisionAndRecall(result):
  print("mAPP: " + str(result["mAPP"]))
  print("mAPN: " + str(result["mAPN"]))
  print("mARP: " + str(result["mARP"]))
  print("mARN: " + str(result["mARN"]))

def toRowVector(vec):
  dim = vec.shape
  rowDim = 1
  for d in dim:
    rowDim = rowDim * d
  return vec.reshape(rowDim)

def detectDuplicatedPic(answer, threshold):
  print("calculating distance...")
  pics = []
  for key in answer:
    for keykey in answer:
      if key == keykey:
        pass
      else:
        distance = answer[key].dot(answer[keykey])/(np.linalg.norm(answer[key])*np.linalg.norm(answer[keykey]))
        flag = False
        if distance >= threshold:
          for pic in pics:
            if key in pic:
              flag = True
              if not keykey in pic:
                pic.append(keykey)
            elif keykey in pic:
              flag = True
              if not key in pic:
                pic.append(key)
          if not flag: 
            pics.append([key,keykey])
  print("done!")
  return pics

def searchNearPic(key, threshold, answer):
  print("calculating distance...")
  pics = {}
  for keykey in answer:
    if key == keykey:
      pass
    else:
      distance = answer[key].dot(answer[keykey])/(np.linalg.norm(answer[key])*np.linalg.norm(answer[keykey]))
      if distance >= threshold:
        pics[keykey] = distance
  print("done!")
  return pics

#select alexnet or googlenet
print(args.image)
if args.model =="alexnet":
  if args.layer == "default":
    args.layer = "fc6"
  answer = alexnetNC.getNeuralCode(args.image, layer=args.layer, gpu=args.gpu) 
  dup = detectDuplicatedPic(answer, 0.99)
elif args.model =="googlenet":
  if args.layer == "default":
    args.layer = "inception_4a/output"
  answer = googlenetNC.getNeuralCode(args.image, layer=args.layer, gpu=args.gpu) 
  vl=vlad.rawVlad(answer,100)
  ivl=vlad.intraVlad(vl)
  feature = {}
  for key in ivl:
    feature[key] = toRowVector(ivl[key])
    feature[key] = feature[key] / np.linalg.norm(feature[key])
  dup = detectDuplicatedPic(feature, 0.9, numberToTry=args.number)
  print(dup)
elif args.model =="illust2vec":
  if args.layer == "default":
    args.layer = "conv5_1"
  answer = illust2vecNC.getNeuralCode(args.image, layer=args.layer, gpu=args.gpu) 
