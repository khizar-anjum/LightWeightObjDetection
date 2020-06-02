# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:42:01 2020

@author: Khizar
"""

import setup_path
import airsim
from keyboard_controller import KeyController

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

c = KeyController(client)

c.listen()