#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import Pycluster as pc
from time import time
from scipy.cluster.vq import kmeans2

"""
 <unblockshaped>
 from here: http://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
 Thanks, unutbu!
"""
def unblockshaped(arr, h, w):
  """
  Return an array of shape (h, w) where
  h * w = arr.size

  If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
  then the returned array preserves the "physical" layout of the sublocks.
  """
  n, nrows, ncols = arr.shape
  return (arr.reshape(h//nrows, -1, nrows, ncols)
          .swapaxes(1,2)
          .reshape(h, w))

def kMeansByPycluster(arr, k, numberToTry=20):
  clusterid, error, nfound = pc.kcluster(arr, nclusters=k, transpose=0,
                                       npass=numberToTry, method='a', dist='e')
  centroids, _ = pc.clustercentroids(arr, clusterid=clusterid)
  return [centroids, clusterid]

def kMeansByScipy(arr, k, threshold=1.0e-05):
  centroids, clusterid = kmeans2(arr, k=k, thresh=threshold, minit='points')
  return [centroids, clusterid]

def normalizeArray(arr):
  answer = np.array(arr[0] / np.linalg.norm(arr[0]))
  for i in range(1,len(arr)):
    if np.linalg.norm(round(arr[i],1)) == 0.0:
        answer = np.vstack([answer, arr[i]])
    else:
      answer = np.vstack([answer, arr[i] / np.linalg.norm(arr[i])])
  return answer

def vlad(dic, k, structure=(14*14,512), numberToTry=20, how="scipy", threshold=1.0e-05):
  keys = []
  newDic = {}
  flag = False
  for key in dic:
    newDic[key] = unblockshaped(dic[key].T, structure[0], structure[1])
    if flag == True:
      dicForkmeans = np.vstack((dicForkmeans,newDic[key]))
    else:
      dicForkmeans = np.array(newDic[key])
      flag = True
    keys.append(key)
  print("k: " + str(k))
  print("number of pictures: " + str(len(dic)))
  print("vector structure for each picture: " + str(structure))
  print("number of vectores for k-means: " + str(len(dicForkmeans)))
  print("start k-means")
  timeMemory = time()
  #execute kMeans
  if how=="scipy":
    result = kMeansByScipy(dicForkmeans, k, threshold=threshold)
  else:
    result = kMeansByPycluster(dicForkmeans, k, numberToTry=numberToTry)
  print("finish k-means")
  print('It took ' + str(int(time() - timeMemory)) + " secondes")
  count = 0
  increment = 0
  #store vlad here
  answer = {}
  for clusterid in result[1]:
    key = keys[count]
    if not answer.has_key(key):
      answer[key] = np.zeros((k,structure(1)))
    answer[key][clusterid] = answer[key][clusterid] + (dic[key][increment] - result[0][clusterid])
    increment = increment + 1
    if increment >= structure(0):
      count = count + 1
      increment = 0
  return answer
