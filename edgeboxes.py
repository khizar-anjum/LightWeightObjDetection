# code adapted from https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/edgeboxes_demo.py
# Author: Khizar Anjum
# Dec 3, 2019
# this code is based on [2] in README.md

import cv2 as cv
import numpy as np
from timeit import default_timer as timer

class edgeboxes():
    #this class is used for getting edgeboxes proposals for an image. 
    #it is based on ximgproc opencv-contrib module. 
    def __init__(self, model, alpha=0.65, beta=0.75, minScore=0.01, maxBoxes=30):
        #init function, sets the initial default parameters.
        self.edge_boxes = cv.ximgproc.createEdgeBoxes()
        self.edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
        self.setAlpha(alpha)
        self.setBeta(beta)
        self.setMinScore(minScore)
        self.setMaxBoxes(maxBoxes)
        self.lasttime = 0
        self.mintime = float("inf")
        
    def setAlpha(self, alpha=0.65):
        self.alpha = alpha
        self.edge_boxes.setAlpha(self.alpha)
        
    def setBeta(self, beta=0.75):
        self.beta = beta 
        self.edge_boxes.setBeta(self.beta)
        
    def setMinScore(self, minScore=0.01):
        self.minScore = minScore
        self.edge_boxes.setMinScore(self.minScore)
        
    def setMaxBoxes(self, maxBoxes=30):
        self.maxBoxes = maxBoxes
        self.edge_boxes.setMaxBoxes(self.maxBoxes)
        
    def getproposals(self, im):
        #this function is used for getting proposal for using in HoG model afterwards
        rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        start = timer()
        edges = self.edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
        orimap = self.edge_detection.computeOrientation(edges)
        edges = self.edge_detection.edgesNms(edges, orimap)
        #boxes = edge_boxes.getBoundingBoxes(edges, orimap)
        boxes, scores = self.edge_boxes.getBoundingBoxes(edges, orimap)
        end = timer()
        self.lasttime = end - start
        if self.lasttime < self.mintime: self.mintime = self.lasttime
        
        return boxes, scores
    
    def plotboxes(self, im, boxes, scores):
        #this function plots boxes and writes their scores on an image, useful for visualization
        if len(boxes) > 0:
            boxes_scores = zip(boxes, scores)
            for b_s in boxes_scores:
                box = b_s[0]
                x, y, w, h = box
                cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)
                score = b_s[1][0]
                cv.putText(im, "{:.2f}".format(score), (x, y), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv.LINE_AA)

        cv.imshow("edgeboxes", im)


