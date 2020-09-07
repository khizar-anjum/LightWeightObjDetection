# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:48:04 2020

@author: khizar
"""

# This file is meant for the implementation of a fully online MDP
# framework as presented inside the paper. 
# Here is the general structure of the framework that I am implementing: 
# 1. Get policy and optimal params obtained via hog_trainer
# 2. Load relevant SVM weights for opres
# 2. Initialize a tracker
# 3. Repeat until end:
#       Request a frame from a frame-server (could be airsim or local folder server)
#       Find the prediction using state from MDP
#       Activate the tracker until the score is > 0.5, otherwise, reactivate MDP

import os
import util
import numpy as np
import pickle
import time
from mdp import mdp_trainer
from tracker import tracker
from hog import hog_predictor
from image_server import image_server

class onlinesim:
    # does the same thing as above
    def __init__(self, folder, param_folder, images = 'local', thresh = 0.5):
        # images parameter describes from where to request images (local or airsim)
        print('Reading Folder Contents...')
        self.folder = folder
        self.param_folder = param_folder
        # reading details about paretoOptimal SVM weights
        print('Reading Details about pareto-Optimal parameters...')
        with open(param_folder + '/opres.pkl', 'rb') as f:
            opres = pickle.load(f)
        #initializing mdp and tracker
        print('Initializing MDP and tracker...')
        self.mdp_ = mdp_trainer(opres)
        self.mdp_.train()
        self.tracker_ = tracker()
        # initializing relevant SVM weights and edgeboxes (both are lumped 
        # inside hog_predictor)
        # we just have to give it proper filename
        print('Initializing edgeboxes, hog-predictors and SVM weights...')
        self.hog_predictors_ = {}
        for k in opres.keys():
            self.hog_predictors_[k] = hog_predictor(os.path.join(param_folder, k + '.pkl'))
        # initialize the image_server
        print('Initializing Image Server...')
        self.im_server = image_server(images, folder = folder)
        # set the threshold for retriggering the MDPs
        self.thresh = thresh

    def start_sim(self):
        overlap_thresh = 0.7
        tracker = False
        state = list(self.hog_predictors_.keys())[0]
        times = []
        accs = []
        while True:
            # request image from server
            image = self.im_server.request()
            # live frame-rate
            if times != []:
                print(f'\n Live Frame Rate: {1/times[-1]}', end='', flush=True)
            else: 
                print('\n Live Frame Rate: 0', end='', flush=True)
            # start the timer
            start = time.time()
            if image is None:
                break 
            if not tracker:
                state = self.mdp_.get_next_state(state)
                boxes, scores = self.hog_predictors_[state].predict(image, 5, overlap_thresh)
                # record relevant information
                end = time.time()
                times.append(end - start)
                accs.append(util.platt(scores[0,0]))
                # If score more than our good-enough threshold, we are good.
                # turn on the tracker
                if util.platt(scores[0,0]) > self.thresh:
                    ret = self.tracker_.init(image, tuple(util.numpy_to_cv2(boxes)[0]))
                    tracker = True
            else:
                ret, boxes = self.tracker_.update(image)
                if ret:
                    b = util.cv2_to_numpy(np.expand_dims(np.array(boxes),0))[0].astype(int)
                    s = self.hog_predictors_[state].get_score_of_tracked_frame(\
                                                    image[b[0]:b[2], b[1]:b[3], :])
                    # record relevant information
                    end = time.time()
                    times.append(end - start)
                    accs.append(util.platt(s[0]))
                    if util.platt(s[0]) > self.thresh:
                        continue
                tracker = False
        return accs, times