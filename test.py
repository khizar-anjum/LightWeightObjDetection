# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:57:32 2019

@author: khizar

This file is for testing purposes only

"""
from itertools import product
import cv2
from hog import hog_multi_trainer
import util
import numpy as np
from mdp import mdp_trainer
import mdptoolbox

params = list(product(*[[500, 1000, 1500, 2000],[(64,64), (32,32), (16,16)], [(16,16), (8,8), (4,4)], [7,8,9]]))
#params = list(product(*[[(32,32), (16,16), (8,8)], [(8,8), (4,4)], [7,8,9]]))
epochs = 5
sign = 'C1'
folder = 'C:/Users/khizar/Documents/JExt20/Experimental-Data-at-BU/KUL dataset/' + sign + '/Seq 1'
param_folder = 'params/' + sign


#%% ONLINE SIM
from onlinesim import onlinesim
o = onlinesim(folder, param_folder, thresh = 0)
accs, times = o.start_sim()

#%% Demonstration of cv2 hogdescriptor computation
img = cv2.imread(folder + '/033507.jpg')
for p in params:
    print(p)
    hog = cv2.HOGDescriptor((64,64), p[0], (8,8), p[1], p[2])
    hog.compute(img)
#%% Demonstration of edgeboxes computation
from edgeboxes import edgeboxes
e = edgeboxes(maxBoxes = 1000)
what = e.getproposals(img)

#%% Trainer, with both hogdescriptor and egdeboxes in the pipeline
hmt = hog_multi_trainer(params, sign, folder, epochs)
hmt.train()

#%% After trained stuff is complete, you get to perform pareto analysis on it. 
from paretoOptimal import paretoOptimalAnalysis
poa = paretoOptimalAnalysis(param_folder, folder)
results = poa.evaluate_params()

#%% plot results for stuff
notop, opres = util.selectparetoOptimal(results)
util.plotResults(notop, opres)

#%% Saving stuff
import pickle
f = open(param_folder + "/opres.pkl","wb")
pickle.dump(opres,f)
f.close()
f = open(param_folder + "/notop.pkl","wb")
pickle.dump(notop,f)
f.close()

#%% testing cuda hog detector
from paretoOptimal import paretoOptimalAnalysis
poa = paretoOptimalAnalysis(param_folder, folder, use_cuda = True)
results = poa.evaluate_params()

#%% Importing results
import pickle 
f = open(param_folder + '/opres.pkl', 'rb')
opres = pickle.load(f)
f.close()
f = open(param_folder + '/notop.pkl', 'rb')
notop = pickle.load(f)
f.close()

#%% Plotting the pareto-Optimal graph
util.plotResults(notop, opres)

#%% RUN THE MDP
w = mdp_trainer(opres)
w.train()
print(w.get_next_state('1000_32_16_9'))

#%% Exploring the policy inside MATLAB code (FUCK THAT)
import pickle
with open('C:/Users/khizar/Documents/JExt20/policy.pkl','rb') as f:
    policy = pickle.load(f)
    
