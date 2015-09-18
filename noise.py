#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from PIL import ImageFilter
from itertools import product
from random import random
import os, re

def rotatePic(inputPath, outputPath, deg=90):
  im = Image.open(inputPath)
  im.rotate(int(deg)).save(outputPath)

#gaussian blur: radius = 2
#http://pillow.readthedocs.org/en/latest/reference/ImageFilter.html
def gaussian(inputPath, outputPath):
  im = Image.open(inputPath)
  im.filter(ImageFilter.GaussianBlur).save(outputPath)

#unsharpmask
#http://pillow.readthedocs.org/en/latest/reference/ImageFilter.html
def unsharpMask(inputPath, outputPath):
  im = Image.open(inputPath)
  im.filter(ImageFilter.UnsharpMask).save(outputPath)

def smooth(inputPath, outputPath):
  im = Image.open(inputPath)
  im.filter(ImageFilter.SMOOTH).save(outputPath)


if __name__ == '__main__':
  base = "test"
  picturePath = [picture for picture in os.listdir(base)
               if re.findall(r"\.png$|\.jpg$|\.JPG$|\.PNG$|\.JPEG$",picture)]
#create folder for pics
  sets = {"90": rotatePic, "G":gaussian, "US":unsharpMask, "S":smooth}
  for key in sets:
    for picture in picturePath:
      if not os.path.exists(base + key):
        os.makedirs(base + key)
      output = base + key + "/" + picture
      inputPath = base + "/" + picture
      sets[key](inputPath, output)
