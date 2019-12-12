# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:05:23 2019

@author: khizar

This python file is intended for pareto-optimal analysis on classifiers obtained 
via permutations of parameters. 
Hence, in the bigger scheme of things, it selects the pareto-optimal parameters
which will become part of state-space for MDP
"""

from hog import hog_predictor
from edgeboxes import edgeboxes
from timeit import default_timer as timer
import numpy as np
from glob import glob
import util

class paretoOptimalAnalysis():
    #assumes that all params files are populated already using hog_multi_trainer
    def __init__(self, sign, folder):
        """
        folder: the folder of a sequence that has ground-truth (can be BU-RU dataset, look into this)
        """
        self.data = self.load_data(folder)
        self.params = self.load_params(sign)
    
    def load_data(self, folder):
        pass
    
    def load_params(self, sign):
        folder = './params/' + sign
        params = []
        for file in glob(folder + '/*.pkl'):
            params.append(util.parse_filename(file))