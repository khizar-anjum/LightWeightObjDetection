# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:51:09 2019

@author: khizar

This file is intended for things related to an MDP algorithm
"""
import mdptoolbox
import numpy as np

class mdp_trainer():
    # This class defines the environment to train object detection for training 
    # an mdp. 
    # Just takes in the optimal results calculated by paretoOptimalAnalysis
    def __init__(self, opres):
        self.opres = opres
        self.available_states = list(self.opres.keys())
        self.num_states = len(opres)
        self.get_rewards()
        self.generate_transitions()
    
    def get_rewards(self):
        self.oprew = {}
        Tmin = np.infty
        # find the minimum time 
        for _,v in self.opres.items():
            if v[1] < Tmin:
                Tmin = v[1]
        # make it the reward thingy
        for k,v in self.opres.items():
            w = v[0]/v[1]
            self.oprew[k] = w*Tmin
        self.R = list(self.oprew.values())
    
    def generate_transitions(self):
        # Calculate the transition matrices
        # since there are two actions, STEP_DOWN(0) and STEP_UP(1)
        # the shape of the transition matrix should be (A,S,S) or (2,S,S)
        self.T = np.zeros((2, self.num_states, self.num_states)) 
        
        self.T[0,0,0] = 1 # you can go any lower
        for s in range(1, self.num_states):
            self.T[0,s,s-1] = 1
        
        self.T[1,-1,-1] = 1 # you can go any higher
        for s in range(self.num_states-2, -1, -1):
            self.T[1,s,s+1] = 1
            
    def train(self):
        # a trainer using the valueIteration of mdptoolbox
        self.vi = mdptoolbox.mdp.ValueIteration(self.T, self.R, 0.9)
        self.vi.run()
        self.policy = self.vi.policy # result is (0, 0, 0)
        
    def get_next_state(self, state):
        # based on the policy: it either 'steps up' or 'steps down' the current 
        # input state. 
        # state should be in the form of a string like '1000_32_16_8', just like 
        # the keys inside the optimal results dictionary
        if state not in self.available_states:
            raise KeyError("State not either in standard format or not a paretoOptimal state")
        
        i = self.available_states.index(state)
        if self.policy[i] == 0: # STEP DOWN
            return self.available_states[i-1 if i > 0 else 0]
        else:
            return self.available_states[i+1 if i < self.num_states-1 else i]