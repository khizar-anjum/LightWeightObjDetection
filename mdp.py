# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:51:09 2019

@author: khizar

This file is intended for things related to an MDP algorithm
"""
import cv2
import numpy as np
from hog import hog_trainer
from itertools import product


class objdetEnv():
    # This class defines the environment to train object detection for training 
    # an mdp. The methods of this environment are loosely defined on the principles
    # of openAI gym environments. However, this class does not inherit from that 
    # and hence is flexible. But it is designed in a way that if you want to port
    # it to an OpenAI gym environment, it should be easy enough.
    def __init__(self, action_space, state_space):
        pass


class mdp_trainer():
    # This class is intended to train an MDP. Parameters are {BB, bsize, csize, nhist}.
    # parameters are first defined as lists and then we define the state-space to be every 
    # conceiveable permutation of them possible. We can export the MDP action-value 
    # table and other values of the states as .npy files
    def __init__(self, BB, bsize, csize, nhist):
        """
        Takes in the following parameters:
            BB: (list) variations of number of bounding boxes for edge boxes e.g. [100, 200, 300]
            bsize: (list) variations of block-size for HOG descriptor e.g. [(32,32),(64,64)]
            csize: (list) variations of cell-size for HOG descriptor e.g. [(16,16),(8,8)]
            nhist: (list) variations of number of histogram bins to use for HOG descriptor
        """
        self.BB = BB
        self.bsize = bsize
        self.csize = csize
        self.nhist = nhist
        all_list = [BB, bsize, csize, nhist]
        self.states = list(product(*all_list)) #all permutations of states
        
    def train():
        pass
