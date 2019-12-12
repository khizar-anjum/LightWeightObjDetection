#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:57:32 2019

@author: khizar

This file is for testing purposes only

"""
from itertools import product
import cv2
from hog import hog_multi_trainer

params = list(product(*[[200, 400, 600, 800],[(64,64), (32,32), (16,16)], [(16,16), (8,8), (4,4)], [7,8,9]]))
#params = list(product(*[[(32,32), (16,16), (8,8)], [(8,8), (4,4),(2,2)], [7,8,9]]))
epochs = 5
signal = 'B5'
folder = '/media/khizar/New Volume/Parul_Research/Experimental Data at BU/KUL dataset/B5/Seq 1'

#%%
"""
#img = cv2.imread(folder + '/033507.jpg')
for p in params:
    print(p)
    hog = cv2.HOGDescriptor((64,64), p[0], (8,8), p[1], p[2])
    hog.compute(img)
"""
#%%
hmt = hog_multi_trainer(params, signal, folder, epochs)
hmt.train()
#%%
"""
from edgeboxes import edgeboxes
e = edgeboxes(maxBoxes = 1000)
what = e.getproposals(img)
"""