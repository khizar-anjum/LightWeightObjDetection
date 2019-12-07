# Author: Khizar Anjum
# Dec 3, 2019
# This file implements HoG model and SVM for detecting street signs. 
# part of this code was inspired from 
# http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-pytorch-2018a.tar.gz

import cv2
import numpy as np
import pandas as pd
import glob
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC

class hog_trainer:
    # this class defines a base hog trainer which will take in proposals from 
    # edge boxes and then train according to images given
    def __init__(self, winSize = (64, 64), blockSize = (16, 16), blockStride = (8, 8), \
                 cellSize = (8, 8), nbins = 9):
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        
        self.hog_desc = cv2.HOGDescriptor (self.winSize, self.blockSize, self.blockStride,\
                 self.cellSize, self.nbins)
        self.svm_weights = np.zeros((self.hog_desc.getDescriptorSize() + 1, 1))
        self.svm = LinearSVC()
        
        self.hog_desc.setSVMDetector(self.svm_weights)
        self.imgs = []
        self.pos_rects = [] #list in list structure for multiple rects in one image
        self.neg_rects = [] #list in list for the same reason
        self.pos_imgs = [] #a list of positive label images, with size equal to winSize
        self.neg_imgs = [] #a list of negative label images, with size equal to winSize
        
        self.overlap_thresh = 0.5 #used to check if boxes overlap enough for detection
        self.n = 5 #top number of detections for hard negative mining
        
    def compute(self, img):
        return self.hog_desc.compute(img)
    
    def populate_data(self, positive = True):
        # used to populate train data using train_rects
        # if train = false, it automatically populate test_rects
        # be careful, removes negative rectangle information when done
        if positive:
            for i, rects in enumerate(self.pos_rects):
                images = []
                for rect in rects:
                    image = self.imgs[i][rect[0]:rect[0]+rect[2],\
                                      rect[1]:rect[1]+rect[3],:]
                    try:
                        images.append(self.compute(cv2.resize(image, self.winSize)))
                    except: #if it cannot be resized, ignore the entry
                        print(rect)
                self.pos_imgs.append(images)
        else:
            for i, rects in enumerate(self.neg_rects):
                for rect in rects:
                    image = self.imgs[i][rect[0]:rect[0]+rect[2],\
                                      rect[1]:rect[1]+rect[3],:]
                    try:
                        self.neg_imgs.append(self.compute(cv2.resize(image, self.winSize)))
                    except: #if it cannot be resized, ignore the entry
                        print("couldn\'t parse",rect)
            self.neg_rects = []
    
    def hard_negative_mine(self, folder, epochs):
        # this function is used to  train svm by hard negative mining
        #initializing svm with default weights of positive sample means
        self.initialize_svm(folder)
        
        for epoch in range(epochs):
            print('Epoch %d, step %d' % ((epoch + 1),1), end='\r')
            # STEP 1: Mine for negative samples
            for i, img in enumerate(self.imgs):
                boxes, scores = self.hog_desc.detectMultiScale(img)
                boxes, scores, _ = self.topn(boxes,scores,self.n)
                boxes, scores, _ = self.nms(boxes,scores)
                #cv2 indexing is different from numpy indexing
                boxes[:,[0,1]] = boxes[:,[1,0]] 
                self.neg_rects.append(boxes.tolist())
            print('Epoch %d, step %d' % ((epoch + 1),2), end='\r')    
            # STEP 2: Add those samples into dataset
            self.populate_data(False)
            print('Epoch %d, step %d' % ((epoch + 1),3), end='\r')
            # STEP 3: Prepare data
            X, y = self.prepare_data()
            print('Epoch %d, step %d' % ((epoch + 1),4), end='\r')
            # STEP 4: train the svm
            self.train_svm(X, y)
        print('Training successfully finished after %d epochs'%epochs)
        
    def nms(self, boxes, scores):
        "Return a tensor of boolean values with True for the boxes to retain"
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]
        n = len(boxes)
        scores_ = scores.copy()
        retain = np.zeros(n).astype(bool)
        minf = float('-inf')
        while True:
            best = np.amax(scores_, 0)
            index = np.argmax(scores_,0)
            if best <= float('-inf'):
                return boxes[retain], scores[retain], retain
            retain[index] = 1
            collision = (self.box_overlap(np.squeeze(boxes[index]), boxes) > \
                         self.overlap_thresh).reshape(-1)
            scores_= np.where(np.expand_dims(collision,axis=1), minf, scores_)    
            
    def box_overlap(self, boxes1, boxes2, measure='iou'):
        """Compute the intersection over union of bounding boxes
    
        Arguments:
            boxes1 {torch.Tensor} -- N1 x 4 tensor with [x0,y0,x1,y1] for N boxes.
                                     For one box, a 4 tensor is also supported.
            boxes2 {torch.Tensor} -- N2 x 4 tensor.
    
        Returns:
            torch.Tensor -- N1 x N2 tensor with the IoU overlaps.
        """
        boxes1 = boxes1.reshape(-1,1,4)
        boxes2 = boxes2.reshape(1,-1,4)
        areas1 = np.prod(boxes1[:,:,:2] - boxes1[:,:,2:], 2)
        areas2 = np.prod(boxes2[:,:,:2] - boxes2[:,:,2:], 2)
    
        max_ = np.maximum(boxes1[:,:,:2], boxes2[:,:,:2])
        min_ = np.minimum(boxes1[:,:,2:], boxes2[:,:,2:])
        intersections = np.prod(np.clip(min_ - max_, 0, float('inf')), 2)
    
        overlaps = intersections / (areas1 + areas2 - intersections)
        return overlaps
    
    def topn(self, boxes, scores, n):
        "Sort the boexes and return the top n"
        n = min(n, len(boxes))
        perm = np.argsort(scores,axis=0)[::-1]
        scores = scores[np.squeeze(perm),:]
        perm = perm[:n]
        scores = scores[:n]
        boxes = boxes[np.squeeze(perm),:]
        return boxes, scores, perm
        
    def initialize_svm(self, folder):
        #used to initialize the weights for svm
        self.read_imgs(folder)
        self.populate_data(True)
        # no need to call it for negative images yet
        
        #training svm with only positive examples to initialize weights
        X, y = self.prepare_data()
        self.train_svm(X,y)
                
            
    def train_svm(self, X, y):
        # fits the svm and updates the svm weights as well as cv2 hogdescriptor 
        # weights
        self.svm.fit(X, y)
        self.svm_weights = np.hstack((np.squeeze(self.svm.coef_),self.svm.intercept_))
        self.hog_desc.setSVMDetector(self.svm_weights)
            
    def prepare_data(self):
        posX = np.array(self.pos_imgs)
        posY = np.ones((posX.shape[0]))
        if self.neg_imgs != []:
            negX = np.expand_dims(np.array(self.neg_imgs),axis=1)
            negY = np.zeros((negX.shape[0]))
            
        else:
            negX = self.compute(np.random.randint(0,high=225,\
                                size=self.winSize + (3,)).astype('uint8'))
            negX = np.expand_dims(np.expand_dims(negX,axis=0),axis=0)
            negY = np.zeros((1,))
        X = np.vstack((posX, negX))
        Y = np.hstack((posY, negY))
        return np.squeeze(X), np.squeeze(Y)
                

        
    def read_imgs(self, folder):
        textfile = glob.glob(folder+'/*.txt')[0]
        df = pd.read_csv(textfile, names=['name','x','y','w','h'],sep=' ')
        df = df.dropna()
        df = df.set_index('name')
        df = df.reindex(columns=['y','x','w','h']) #taking care of matlab quirk
        for file in set(df.index):
            self.imgs.append(cv2.imread(folder + '/' + file))
            self.pos_rects.append([df.loc[file].values.astype(int).tolist()])
        
    def plot_box(self, box, color='y'):
        #imported from lab.py from category detection 2019 pytorch version
        r1 = Rectangle(box[:2],
                       box[2], box[3],
                       facecolor='none', linestyle='solid',
                       edgecolor=color, linewidth=3)
        r2 = Rectangle(box[:2],
                       box[2], box[3],
                       facecolor='none', linestyle='solid',
                       edgecolor='k', linewidth=5)
        plt.gca().add_patch(r2)
        plt.gca().add_patch(r1)
