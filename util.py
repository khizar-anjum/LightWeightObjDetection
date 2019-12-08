# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:02:17 2019

@author: khizar

File for the purpose of saving functions that are used often
"""
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt

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
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = 'â–ˆ', printEnd = "\r"):
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