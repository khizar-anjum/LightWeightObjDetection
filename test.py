# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:57:32 2019

@author: khizar

This file is for testing purposes only

"""
from itertools import product
import cv2
from hog import hog_multi_trainer
import numpy as np

params = list(product(*[[500, 1000, 1500, 2000],[(64,64), (32,32), (16,16)], [(16,16), (8,8), (4,4)], [7,8,9]]))
#params = list(product(*[[(32,32), (16,16), (8,8)], [(8,8), (4,4),(2,2)], [7,8,9]]))
epochs = 5
signal = 'C1'
folder = '/media/khizar/New Volume/Parul_Research/Experimental Data at BU/KUL dataset/C1/Seq 1'
param_folder = 'params/C1'

#%%
"""
#img = cv2.imread(folder + '/033507.jpg')
for p in params:
    print(p)
    hog = cv2.HOGDescriptor((64,64), p[0], (8,8), p[1], p[2])
    hog.compute(img)
"""
#%%
"""
from edgeboxes import edgeboxes
e = edgeboxes(maxBoxes = 1000)
what = e.getproposals(img)
"""
#%%
#from paretoOptimal import paretoOptimalAnalysis
#poa = paretoOptimalAnalysis(param_folder, folder)
#results = poa.evaluate_params()
#import util
#notop, opres = util.selectparetoOptimal(results)
#util.plotResults(notop, opres)


#%% testing cuda hog detector
from paretoOptimal import paretoOptimalAnalysis
poa = paretoOptimalAnalysis(param_folder, folder, use_cuda = True)
results = poa.evaluate_params()
