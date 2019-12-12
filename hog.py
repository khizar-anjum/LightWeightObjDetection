# Author: Khizar Anjum
# Dec 3, 2019
# This file implements HoG model and SVM for detecting street signs. 
# part of this code was inspired from 
# http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-pytorch-2018a.tar.gz

import cv2
import numpy as np
import pandas as pd
import glob
from sklearn.svm import LinearSVC
from edgeboxes import edgeboxes
import util
import pickle

class hog_trainer:
    # this class defines a base hog trainer which takes in a folder (str) as an input
    # and then trains an svm classifier based on given hog parameters using the 
    # negative hard mining technique (this svm is used in cv2.HOGDescriptor.detectMultiScale)
    # normal usage to train a folder full of images is given by:
    # >> hog_t = hog_trainer(whatever parameters you want to initialize hog with)
    # >> hog_t.hard_negative_mine(folder, epochs)
    # after that, you export the svm weights from hog_t.svm_weights
    # Very slow, dont use on a folder with more than 100 images
    # Folder should have a text file of annotated rectangle patches
    def __init__(self, winSize = (64, 64), blockSize = (16, 16), blockStride = (8, 8), \
                 cellSize = (8, 8), nbins = 9, overlap_thresh = 0.5, n = 5, maxBoxes = 30):
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        self.maxBoxes = maxBoxes
        
        self.hog_desc = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride,\
                 self.cellSize, self.nbins)
        self.svm_weights = np.zeros((self.hog_desc.getDescriptorSize() + 1, 1))
        self.edgebox = edgeboxes(maxBoxes = self.maxBoxes)
        self.svm = LinearSVC()
        
        self.hog_desc.setSVMDetector(self.svm_weights)
        self.imgs = []
        self.pos_rects = [] #list in list structure for multiple rects in one image
        self.neg_rects = [] #list in list for the same reason
        self.pos_imgs = [] #a list of positive label images, with size equal to winSize
        self.neg_imgs = [] #a list of negative label images, with size equal to winSize
        
        self.overlap_thresh = overlap_thresh #used to check if boxes overlap enough for detection
        self.n = n #top number of detections for hard negative mining
        
    def compute(self, img):
        return self.hog_desc.compute(img)
    
    def populate_data(self, positive = True):
        # used to populate train data using train_rects
        # if train = false, it automatically populate test_rects
        # be careful, removes negative rectangle information when done
        if positive:
            self.pos_imgs = []
            for i, rects in enumerate(self.pos_rects):
                images = []
                for rect in rects:
                    image = self.imgs[i][rect[0]:rect[2],\
                                      rect[1]:rect[3],:]
                    try:
                        images.append(self.compute(cv2.resize(image, self.winSize)))
                    except: #if it cannot be resized, ignore the entry
                        print("couldn\'t parse",rect)
                self.pos_imgs.append(images)
        else:
            for i, rects in enumerate(self.neg_rects):
                for rect in rects:
                    image = self.imgs[i][rect[0]:rect[2],\
                                      rect[1]:rect[3],:]
                    try:
                        self.neg_imgs.append(self.compute(cv2.resize(image, self.winSize)))
                    except: #if it cannot be resized, ignore the entry
                        print("couldn\'t parse",rect)
            self.neg_rects = []
    
    def hard_negative_mine(self, folder, epochs):
        # this function is used to  train svm by hard negative mining
        #initializing svm with default weights of positive sample means
        self.initialize_svm(folder)
        util.printProgressBar(0, epochs, prefix = 'Starting', suffix = 'complete')
        for epoch in range(epochs):
            # STEP 1: Mine for negative samples
            for i, img in enumerate(self.imgs):
                boxes, scores = self.edgebox.getproposals(img)
                if not isinstance(boxes,tuple): #because if boxes is tuple, its empty
                    boxes = util.cv2_to_numpy(boxes)
                    boxes, scores = self.filter_negsamples(boxes, scores,i)
                    if(boxes.shape[0] > self.n): #only do it if more than n samples
                        boxes, scores, _ = self.topn(boxes,scores,self.n)
                    if(boxes.shape[0] > 0): #only do it some boxes survive the journey
                        boxes, scores, _ = self.nms(boxes,scores)
                        self.neg_rects.append(boxes.tolist())
            #    else:
            #        print("no boxes detected!")
                
            # STEP 2: Add those samples into dataset
            self.populate_data(True)
            self.populate_data(False)
            
            # STEP 3: Prepare data
            X, y = self.prepare_data()
            
            # STEP 4: train the svm
            self.train_svm(X, y)
            util.printProgressBar(epoch+1, epochs, prefix = 'Epoch %d'%(epoch+1), suffix = 'complete')
        print('Training successfully finished after %d epochs'%epochs)
        
    def filter_negsamples(self, boxes, scores, i):
        pos_boxes = np.squeeze(np.array(self.pos_rects[i]))
        overlaps = self.box_overlap(pos_boxes, boxes)
        thresh_boxes = np.sum(overlaps > self.overlap_thresh, axis=0) > 0
        if any(thresh_boxes):
            self.pos_rects[i] = self.pos_rects[i] + boxes[thresh_boxes].tolist()
        return boxes[~thresh_boxes], scores[~thresh_boxes] #picking boxes which do not overlap
        
    def nms(self, boxes, scores):
        "Return a tensor of boolean values with True for the boxes to retain"
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
        #just flattening a list of lists
        posX = np.array([item for sublist in self.pos_imgs for item in sublist])
        posY = np.ones((posX.shape[0]))
        if self.neg_imgs != []:
            negX = np.array(self.neg_imgs)
            negY = np.zeros((negX.shape[0]))
            
        else:
            negX = self.compute(np.random.randint(0,high=225,\
                                size=self.winSize + (3,)).astype('uint8'))
            negX = np.expand_dims(negX,axis=0)
            negY = np.zeros((1,))
        X = np.vstack((posX, negX))
        Y = np.hstack((posY, negY))
        return np.squeeze(X), np.squeeze(Y)
                
    def read_imgs(self, folder):
        textfile = glob.glob(folder+'/*.txt')[0]
        df = pd.read_csv(textfile, names=['name','x','y','w','h'],sep=' ')
        df = df.dropna()
        df = df.set_index('name')
        df['w'] = df['x'] + df['w']
        df['h'] = df['y'] + df['h']
        df = df.reindex(columns=['y','x','h','w']) #taking care of matlab quirk
        for file in set(df.index):
            self.imgs.append(cv2.imread(folder + '/' + file))
            self.pos_rects.append([df.loc[file].values.astype(int).tolist()])

class hog_multi_trainer():
    # this class is used to train multiple hog models (svm weights basically)
    # and then save them in files so that they are easy to read afterwards.
    # it saves them in ./hog/{sign}/{filnameAccordingToConvention.pkl}
    def __init__(self, params, sign, folder, epochs):
        """
        params: a list containing [maxBoxes, blockSize, cellSize, nbins] in this order
        sign: the traffic sign from KUL dataset we are working with
        folder: the folder which contains all the training images
        epochs: number of epochs to run a trainer for
        """
        self.params = params
        self.sign = sign
        self.folder = folder
        self.epochs = epochs
        
    def train(self):
        for p in self.params:
            self.maxBoxes, self.blockSize, self.cellSize, self.nbins = p
            hog = hog_trainer(blockSize = self.blockSize, maxBoxes = self.maxBoxes,\
                             cellSize = self.cellSize, nbins = self.nbins)
            hog.hard_negative_mine(self.folder, self.epochs)
            self.save_config(hog.svm_weights)
        
    def save_config(self, weights):
        # uses pickle to save all the information used to train the model
        # filename is chosen according to the configuration of params
        filename = util.params_to_filename([self.sign, self.maxBoxes, self.blockSize, \
                                                self.cellSize, self.nbins])
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(weights, f)

 
class hog_predictor():
    # this class is used to infer if a given proposal coming from edgeboxes 
    # proposal function is an object of interest or not. Basically, we just see 
    # if a incoming proposals score positive on the SVM plane or not after
    # computing their hog features
    # class that defines a edgeboxes->hog->svm predictor
    def __init__(self, filename):
        self.sign, self.maxBoxes, self.blockSize, self.cellSize, \
                self.nbins = util.parse_filename(filename)
        self.load_config(filename)
        
    def info(self):
        return {self.edgebox.maxBoxes, self.hog.blockSize, self.hog.cellSize, self.nbins}
        
    def predict(self, image):
        p = self.edgebox.getproposals(image)
        scores = self.infer(p)
        return p, scores
    
    def load_config(self, filename):
        # uses pickle to load all the tunable parameter information
        with open(filename,'rb') as f:
            self.svm_weights = pickle.load(f)
        self.edgebox = edgeboxes(maxBoxes = self.maxBoxes)
        self.hog_desc = cv2.HOGDescriptor((64,64), self.blockSize, (8,8),\
                                          self.cellSize, self.nbins)        
        
    def infer(self, proposals):
        # expects proposals in a list (image patches)
        scores = []
        for p in proposals:
            hogf = self.hog_desc.compute(cv2.resize(p,self.winSize))
            s = self.svm_weights[:-1] @ hogf + self.svm_weights[-1]
            scores.append(s)
        return scores
