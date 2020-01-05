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
from timeit import default_timer as timer
import numpy as np
import util

class paretoOptimalAnalysis():
    "used to perform speed-up/AUC analysis to select paretoOptimal params"
    def __init__(self, param_folder, data_folder, overlap_thresh = 0.5, n = 100, use_cuda = False):
        """
        data_folder: the folder of a sequence that has ground-truth (can be BU-RU dataset, look into this)
        param_folder: folder in which all params files are populated already using hog_multi_trainer
        """
        self.imgs, self.pos_rects = util.read_imgs(data_folder)
        # self.params contains all the combinations of [BB,bsize,csize,nbins,weights]
        # list in list structure basically
        self.sign, self.params = util.load_params(param_folder)
        
        self.overlap_thresh = overlap_thresh
        self.n = n
        self.use_cuda = use_cuda
        
    def evaluate_params(self):
        eval_results = {}
        for param in self.params:
            hogInfer = hog_predictor(util.params_to_filename([self.sign] + list(param)),\
                                     use_cuda = self.use_cuda)
            results = self.evaluate_model(hogInfer)
            _, _, overall_auc = util.pr(results['labels'], results['scores'], \
                                  misses = results['misses'], plot=False)
            overall_time = np.array(results['times']).mean()
            f = str(param[0]) + '_' + str(param[1][0]) + '_' +\
                    str(param[2][0]) + '_' + str(param[3])
            eval_results[f] = (overall_auc, overall_time)
            print(f'{f} have a time of {overall_time} and AUC of {overall_auc}')
        return eval_results
        
    def eval_detections(self,gt_boxes, boxes, threshold=0.5, plot=False, gt_difficult=None):
        if len(gt_boxes) == 0:
            return {
                'gt_to_box': [],
                'box_to_gt': np.array([-1]*len(boxes)),
                'labels': np.array([-1]*len(boxes)),
                'misses': 0,
            }
        
        # Compute the overlap between ground-truth boxes and detected ones.
        overlaps = util.box_overlap(boxes, gt_boxes)

        # Match each box to a gt box.
        box_to_gt = np.argmax(overlaps, axis=0)
        overlaps = np.max(overlaps, axis=1)
        matched = overlaps > threshold
        labels = -np.ones(len(boxes))
        labels[matched] = +1

        # Discount the boxes that match difficult gts
        if gt_difficult is not None:
            discounted = matched & gt_difficult[box_to_gt]
            matched &= discounted ^ 1 # logic negation as XOR
            labels[discounted.nonzero()] = 0

        misses = 0
        gt_to_box = np.full((len(gt_boxes),), -1, dtype=np.int64)
        for i in range(len(gt_boxes)):
            if gt_difficult is not None and gt_difficult[i]:
                continue
            j = np.nonzero((box_to_gt == i) & matched)
            j = j[0]
            if len(j) == 0:
                misses += 1
            else:
                gt_to_box[i] = j[0]
                labels[j[1:]] = -1
            matched[j] = 0
        
        if plot:
            for box in gt_boxes:
                util.plot_box(box, color='y')
            for box, label in reversed(list(zip(boxes, labels))):
                util.plot_box(box, color='g' if label > 0 else 'r')

        return {
            'gt_to_box': gt_to_box,
            'box_to_gt': box_to_gt,
            'labels': labels,
            'misses': misses,
        }
        
    def evaluate_model(self, hogInfer):
        "Evaluate the model by looping over imgs loaded from the folder"
        # Loop over all images in the dataset
        all_labels = []
        all_scores = []
        all_times = []
        negs = []
        misses = 0
    
        for t, image in enumerate(self.imgs):         
            # Pick all the gt boxes in the selected image
            gt_boxes = np.array(self.pos_rects[t])
            start = timer()
            # Run the detector
            boxes, scores = hogInfer.predict(image, self.n, self.overlap_thresh)
            end = timer()
            thistime = end - start
            # Evaluate the detector and plot the results
            results = self.eval_detections(gt_boxes, boxes)
            all_labels.append(results['labels'])
            all_scores.append(scores)
            all_times.append(thistime)
            misses += results['misses']
    
            # Compute the per-image AP
            _, _, ap = util.pr(results['labels'], scores, misses=results['misses'], plot=False)
            print(f"Evaluating on image {t+1:3d} of {len(self.imgs):3d}: AP: {ap*100:6.1f}%")
            
        return {
            'labels' : np.concatenate(all_labels, axis=0),
            'scores' : np.concatenate(all_scores, axis=0),
            'misses' : misses,
            'negatives' : negs,
            'times' : all_times
        }
        
    
    
    
    