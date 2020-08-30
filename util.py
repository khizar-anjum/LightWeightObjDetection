# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:02:17 2019

@author: khizar

File for the purpose of saving functions that are used often
"""
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from glob import glob
import pandas as pd
import cv2
import numpy as np
import pickle
"""
cv2/matplotlib box representation format: (x,y,w,h) where x is along width
    and y is along height of the image with origin at the top-left corner
    
numpy box representation format: (x0,y0,x1,y1) where x is along height and y 
    is along width of the image with origin at the top-left corner
"""


def cv2_to_numpy(cv2_boxes):
    # used to fix cv2 indexing into numpy indexing of images
    # basically x and y are flipped in numpy, x is along height and y is along width
    # and convert (y,x,h,w) into (x0,y0,x1,y1)
    # cv2 indexing is different from numpy indexing
    # expects a numpy array of boxes of size (N x 4)
    boxes = cv2_boxes.copy()
    boxes[:,[0,1]] = boxes[:,[1,0]]
    boxes[:,[2,3]] = boxes[:,[3,2]]
    boxes[:,2] = boxes[:,0] + boxes[:,2]
    boxes[:,3] = boxes[:,1] + boxes[:,3]
    return boxes

def numpy_to_cv2(np_boxes):
    # used to convert numpy indexing into cv2 indexing
    # tbh, cv2 indexing is the sane one. 
    # converts (x0,y0,x1,y1) into (y, x, h, w)
    # plt also uses cv2 type indexing where x is along the width and y is along height
    # expects a numpy array of boxes of size (N x 4)
    boxes = np_boxes.copy()
    boxes[:,[0,1]] = boxes[:,[1,0]]
    boxes[:,[2,3]] = boxes[:,[3,2]]
    boxes[:,2] = boxes[:,2] - boxes[:,0]
    boxes[:,3] = boxes[:,3] - boxes[:,1]
    return boxes
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
def plot_box(box, color='y'):
    #imported from lab.py from category detection 2019 pytorch version
    #input box must be in cv2/matplotlib format
    #expects a list of length 4
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

def plot_boxes(boxes, format_ = 'cv2'):
    #you can give in which format the boxes are already
    #format_ input expects 'cv2' or 'numpy' as inputs
    #takes in multiple boxes to plot them on an already open image
    #expects a numpy array of the format (N x 4)
    assert (format_ == 'cv2' or format_ == 'numpy'), "only 'cv2' or 'numpy' are valid arguments"
    my_boxes = boxes.copy()
    if(format_ == 'numpy'): my_boxes = numpy_to_cv2(my_boxes)
    for box in my_boxes:
        plot_box(box.tolist())
    
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def params_to_filename(params):
    # takes in params in a list
    # the order should always be [sign, maxBoxes, blockSize, cellSize, nbins, svm_weights]
    # cellsize and blockSize should be tuples i.e. (32,32) or (64,64)
    filename = str(params[1]) + '_' + str(params[2][0]) + '_' +\
                    str(params[3][0]) + '_' + str(params[4]) + '.pkl'
    filename = 'params/' + params[0] + '/' + filename
    return filename

def parse_filename(filename):
    filename = filename.replace('\\', '/')
    #print(filename)
    _, sign, params = filename.split('/')
    params, _ = params.split('.')
    BB, bsize, csize, nbins = [int(x) for x in params.split('_')]
    return sign, (BB, (bsize, bsize), (csize, csize), nbins)

def read_imgs(folder):
    imgs = []
    pos_rects = []
    textfile = glob(folder+'/*.txt')[0]
    df = pd.read_csv(textfile, names=['name','x','y','w','h'],sep=' ')
    df = df.dropna()
    df = df.set_index('name')
    df['w'] = df['x'] + df['w']
    df['h'] = df['y'] + df['h']
    df = df.reindex(columns=['y','x','h','w']) #taking care of matlab quirk
    for file in set(df.index):
        imgs.append(cv2.imread(folder + '/' + file))
        pos_rects.append([df.loc[file].values.astype(int).tolist()])
    return imgs, pos_rects

def box_overlap(boxes1, boxes2, measure='iou'):
    """Compute the intersection over union of bounding boxes

    Arguments:
        boxes1 {numpy array} -- N1 x 4 tensor with [x0,y0,x1,y1] for N boxes.
                                 
        boxes2 {numpy array} -- N2 x 4 tensor.

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

def nms(boxes, scores, overlap_thresh):
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
        collision = (box_overlap(np.squeeze(boxes[index]), boxes) > \
                     overlap_thresh).reshape(-1)
        scores_= np.where(np.expand_dims(collision,axis=1), minf, scores_)
        
def topn(boxes, scores, n):
    "Sort the boxes and return the top n"
    n = min(n, len(boxes))
    perm = np.argsort(scores,axis=0)[::-1]
    scores = scores[np.squeeze(perm),:]
    perm = perm[:n]
    scores = scores[:n]
    boxes = boxes[np.squeeze(perm),:]
    return boxes, scores, perm

def load_params(folder):
    "used to read files from a params folder"
    params = []
    for file in glob(folder + '/*.pkl'):
        sign, param = parse_filename(file)
        with open(file,'rb') as f:
            param = param + (pickle.load(f),)
        params.append(param)
    return sign, params

def sort(scores, descending = False):
    indices = np.argsort(scores,axis=0)
    scores = np.sort(scores,axis=0)
    if descending:
        indices = np.flip(indices)
        scores = np.flip(scores)
    return scores, indices

def pr(labels, scores, misses=0, plot=True):
    "Plot the precision-recall curve."
    scores, perm = sort(scores, descending=True)
    labels = labels[np.squeeze(perm)]
    tp = (labels > 0).astype(np.float32)
    ttp = np.cumsum(tp, axis=0)
    precision = np.divide(ttp , np.arange(1, len(tp)+1, dtype=np.float32))
    recall = ttp / np.clip(tp.sum() + misses, a_min=1, a_max=None)
    # Labels may contain no positive labels (perhaps because misses>0)
    # which would case mean() to nan
    ap = precision[tp > 0]
    ap = ap.mean() if len(ap) > 0 else 0
    if plot:
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0,1.01)
        plt.ylim(0,1.01)
    return precision, recall, ap

def plotResults(results, pOptimal):
    "plots results generated by paretoOptimal.evaluate_params"
    for i, res in enumerate([results, pOptimal]):
        vals = [v for _, v in res.items()]
        vals = np.array(vals)
        X, y = vals[:,0], vals[:,1]
        y = y/np.min(y)
        if(i == 0): plt.plot(X, y,'bo')
        else: 
            sort = list(np.argsort(np.array(X)))
            X = [X[i] for i in reversed(sort)]
            y = [y[i] for i in reversed(sort)]
            plt.plot(X, y,'ro-')
    plt.xlabel('Area under curve (AUC)')
    plt.ylabel('Speedup')
    plt.show()
    
def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] > scores[i]) and any(scores[j] >= scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[np.invert(pareto_front)], population_ids[pareto_front]
    
def selectparetoOptimal(results):
    "results paretoOptimal params from the ones generated by paretoOptimal.evaluate_params"
    vals = [v for _, v in results.items()]
    rresults = {k:v for v,k in results.items()}
    vals = np.array(vals)
    X, y = vals[:,0], vals[:,1] #x is auc, y is speedup
    npareto, pareto = identify_pareto(np.array([X,y]).T)

    pOptimal = [(X[i], y[i]) for i in pareto]
    notOptim = [(X[i], y[i]) for i in npareto]
    opres = {rresults[p]:p for p in pOptimal}
    notop = {rresults[p]:p for p in notOptim}
    return notop, opres