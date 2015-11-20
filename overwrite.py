#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, time

def overwrite(sen):
  sys.stdout.write("\r" + sen)
  sys.stdout.flush()

def bar(a, b):
  num = int(float(a)/b*100)
  numSharp = int(num / 5)
  numSpace = int(20 - numSharp)
  if num < 10:
    return " " + str(num) + "%: " +  "[" + "#" * numSharp + " " * numSpace + "]"
  else:
    return str(num) + "%: " +  "[" + "#" * numSharp + " " * numSpace + "]"
