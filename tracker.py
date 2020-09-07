# Author: Khizar Anjum
# Dec 3, 2019
# This file implements a wrapper class for trackers in opencv

import cv2

class tracker():
    #depending on whichever you choose, this provides a nice wrapper
    def __init__(self, tracker_type = 'KCF'):
        if tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        else:
            raise KeyError('Tracker type not found')
            
    def init(self, frame, bbox0):
        return self.tracker.init(frame, bbox0)
        
    def update(self, frame):
        return self.tracker.update(frame)