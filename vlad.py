#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import Pycluster as pc
from time import time
import random
from scipy.cluster.vq import kmeans2
from scipy.linalg import sqrtm
from scipy.linalg import pinv2
from scipy.linalg import logm

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

def calcDeg(arr1,arr2):
  return np.arccos(arr1.dot(arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))

def intraVlad(dic):
  answer = {}
  for key in dic:
    tmp = []
    for arr in dic[key]:
      if np.linalg.norm(arr)==0.0:
        if len(tmp)!=0:
          tmp=np.vstack((tmp,arr))
        else:
          tmp=np.array(arr)
      else:
        if len(tmp)!=0:
          tmp=np.vstack((tmp,arr/np.linalg.norm(arr)))
        else:
          tmp=np.array(arr/np.linalg.norm(arr))
    answer[key] = tmp
  return answer
  

def rawVlad(dic, k, structure=(14*14,512), numberToTry=20, how="scipy", threshold=1.0e-05):
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
      answer[key] = np.zeros((k,structure[1]))
    answer[key][clusterid] = answer[key][clusterid] + (newDic[key][increment] - result[0][clusterid])
    increment = increment + 1
    if increment >= structure[0]:
      count = count + 1
      increment = 0
  return answer

def rawgVlad(dic, k, structure=(14*14,512), numberToTry=20, how="scipy", threshold=1.0e-05):
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
      answer[key] = np.zeros((k,structure[1]))
    deg = calcDeg(newDic[key][increment],  result[0][clusterid])
    answer[key][clusterid] = answer[key][clusterid] + (newDic[key][increment]**deg - result[0][clusterid])
    increment = increment + 1
    if increment >= structure[0]:
      count = count + 1
      increment = 0
  return answer

def infToSmallNum(arr, inflation=1.0e-15):
  answer = np.zeros(arr.shape, dtype="float")
  for i in xrange(len(arr)):
    answer[i] = float(arr[i])
    if answer[i] == 0.0:
      answer[i] = inflation
  return answer

def geodesicDistanceOnSPD(x, y):
  if len(x.shape)!=1:
    sq = sqrtm(x)
    invsq = pinv2(sq)
    F = np.dot(np.dot(invsq, y), invsq)
    return np.linalg.norm(logm(F))
  else:
    x = infToSmallNum(x)
    y = infToSmallNum(y)
    sq = x**0.5
    invsq = 1.0 / sq
    F = invsq * y * invsq
    return np.linalg.norm(np.log(F))

def derivativeSquareOfGeodesicOnSPD(x,y):
  if len(x.shape)!=1:
    sq = sqrtm(x)
    invsq = pinv2(sq)
    F = np.dot(np.dot(invsq, y), invsq)
    return 2*np.dot(np.dot(sq,logm(F),sq))
  else:
    x = infToSmallNum(x)
    y = infToSmallNum(y)
    sq = x**0.5
    invsq = 1.0 / sq
    F = invsq * y * invsq
    return 2*sq*np.log(F)*sq

def eStepGeodesicOnSPD(centroid, vec):
  return geodesicDistanceOnSPD(centroid, vec) **2

def calcMeanOnSPD(p, q):
  if len(p.shape)!=1:
    sq = sqrtm(p)
    invsq = pinv2(sq)
    F = sqrtm(np.dot(np.dot(sq, q), sq))
    return np.dot(np.dot(invsq,F),invsq)
  else:
    p = infToSmallNum(p)
    q = infToSmallNum(q)
    sq = p**0.5
    invsq = 1.0 / sq
    F = (sq * q * sq) **0.5
    return invsq * F * invsq

def mStepGeodesicOnSPD(vec, centroidId, k):
  answer = np.zeros((k,len(vec[0])))
  p = np.zeros((k,len(vec[0])))
  q = np.zeros((k,len(vec[0])))
  if len(vec[0].shape)!=1:
    for i in enumerate(centroidId):
      p[i[1]] = p[i[1]] + pinv2(vec[i[0]])
      q[i[1]] = q[i[1]] + vec[i[0]]
  else:
    for i in enumerate(centroidId):
      p[i[1]] = p[i[1]] + 1.0 / infToSmallNum(vec[i[0]])
      q[i[1]] = q[i[1]] + vec[i[0]]
  for i in xrange(len(p)):
    answer[i] = calcMeanOnSPD(p[i],q[i])
  return answer

#assign centroid
def eStep(vectores, centroid, distance):
  centroidId = []
  for vec in vectores: 
    tmpVal = 9999999
    tmpIndex = -1
    for i in xrange(len(centroid)): 
      dis = distance(centroid[i], vec)
      if dis < tmpVal:
        tmpVal = dis
        tmpIndex = i
    centroidId.append(tmpIndex)
  return centroidId

#calc cluster center
def mStep(vectores, centroidId, calcCenter, k):
  return calcCenter(vectores, centroidId, k)

def initRandomly(arr, k):
  return random.sample(xrange(len(arr)),k)

def kmeansForManifold(arr, k, distance, calcCenter, iteration=10):
  centroid = arr[initRandomly(arr,k)]
  for i in xrange(iteration):
    centroidId = eStep(arr, centroid, distance)
    centroid = mStep(arr, centroidId, calcCenter, k)
  centroidId = eStep(arr, centroid, distance)
  return [centroid, centroidId]

def rawrVlad(dic, k, structure=(14*14,512), iteration=20):
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
  result = kmeansForManifold(dicForkmeans,k,eStepGeodesicOnSPD,mStepGeodesicOnSPD,iteration=20)
  print("finish k-means")
  print('It took ' + str(int(time() - timeMemory)) + " secondes")
  count = 0
  increment = 0
  #store vlad here
  answer = {}
  for clusterid in result[1]:
    key = keys[count]
    if not answer.has_key(key):
      answer[key] = np.zeros((k,structure[1]))
    deri = derivativeSquareOfGeodesicOnSPD(result[0][clusterid],newDic[key][increment])
    answer[key][clusterid] = answer[key][clusterid] + (
        geodesicDistanceOnSPD(result[0][clusterid],newDic[key][increment]) / np.linalg.norm(deri) * deri)
    increment = increment + 1
    if increment >= structure[0]:
      count = count + 1
      increment = 0
  return answer
