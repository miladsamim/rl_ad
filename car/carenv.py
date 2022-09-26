import numpy as np
import carla
import cv2

class CarEnv:
    def __init__(self, car, carequipment):
        self.car = car
        self.carequipment = carequipment


    def step(self, action, **kwargs):
        pass 

