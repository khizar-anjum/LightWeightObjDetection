# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 20:42:45 2020

@author: khizar
"""
import os
import cv2
import glob
import airsim

class image_server:
    # serves images to the online simulation when requested
    # sends a None response when receives no image
    # images could be 'local' or 'airsim'
    def __init__(self, images = 'local', folder = None):
        self.images = images
        if images == 'local':
            self.ims = glob.glob(os.path.join(folder, '*.jpg'))
        else:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
        self.i = 0
    
    def request(self):
        if self.images == 'local':
            if self.i < len(self.ims)-1: 
                im = cv2.imread(self.ims[self.i])
                self.i += 1
                return im
        else:
            responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.Scene)]) #scene vision image in png format
            return responses[0]
        return None
        