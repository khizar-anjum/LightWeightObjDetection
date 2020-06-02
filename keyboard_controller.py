# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:50:06 2020

@author: Khizar
"""

from pynput.keyboard import Key, Listener, KeyCode
import airsim

class KeyController:
    def __init__(self, client):
        self.client = client
        self.vx = 5
        self.vy = 5
        self.vz = 5
        self.yaw_rate = 10
        self.dur = 0.1
    
    def on_press(self,key):
        print('{0} pressed'.format(
            key))
        if key == Key.up:
            #move forward
            self.client.moveByVelocityAsync(self.vx,0,0,self.dur).join()
        elif key == Key.down:
            #move backward
            self.client.moveByVelocityAsync(-self.vx,0,0,self.dur).join()
        elif key == Key.left:
            #move left
            self.client.moveByVelocityAsync(0,self.vy,0,self.dur).join()
        elif key == Key.right:
            #move right
            self.client.moveByVelocityAsync(0,-self.vy,0,self.dur).join()
        elif key == KeyCode.from_char('d'):
            #rotate clockwise
            self.client.rotateByYawRateAsync(self.yaw_rate,self.dur).join()
        elif key == KeyCode.from_char('a'):
            #rotate anti-clockwise
            self.client.rotateByYawRateAsync(-self.yaw_rate,self.dur).join()
        elif key == KeyCode.from_char('w'):
            #move up in z-axis
            self.client.moveByVelocityAsync(0,0,-self.vz,self.dur).join()
        elif key == KeyCode.from_char('s'):
            #move down in z-axis
            self.client.moveByVelocityAsync(0,0,self.vz,self.dur).join()
        elif key == Key.space:
            #takeoff
            self.client.takeoffAsync().join()
        elif key == KeyCode.from_char('m'):
            #reset
            self.client.reset()
        elif key == KeyCode.from_char('n'):
            #land
            self.client.landAsync().join()
    
    def on_release(self,key):
        print('{0} release'.format(
            key))
        if key == Key.esc:
            # Stop listener
            if self.client.getMultirotorState().landed_state == \
                    airsim.LandedState.Landed:
                pass
            else:
                self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            return False
        elif key == Key.up:
            #move forward
            self.client.moveByVelocityAsync(0,0,0,self.dur).join()
        elif key == Key.down:
            #move backward
            self.client.moveByVelocityAsync(0,0,0,self.dur).join()
        elif key == Key.left:
            #move left
            self.client.moveByVelocityAsync(0,0,0,self.dur).join()
        elif key == Key.right:
            #move right
            self.client.moveByVelocityAsync(0,0,0,self.dur).join()
        elif key == KeyCode.from_char('d'):
            #rotate clockwise
            self.client.rotateByYawRateAsync(0,self.dur).join()
        elif key == KeyCode.from_char('a'):
            #rotate anti-clockwise
            self.client.rotateByYawRateAsync(0,self.dur).join()
        elif key == KeyCode.from_char('w'):
            #move up in z-axis
            self.client.moveByVelocityAsync(0,0,0,self.dur).join()
        elif key == KeyCode.from_char('s'):
            #move down in z-axis
            self.client.moveByVelocityAsync(0,0,0,self.dur).join()

    def listen(self):
        # Collect events until released
        with Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()
        
        
#c = Controller(1a)
#c.listen()