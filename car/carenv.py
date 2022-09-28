import numpy as np
import carla
import cv2

class CarEnv:
    def __init__(self, ego_vehicle, carequipment, env_setup, host='localhost', port=2000):
        client = carla.Client('localhost', 2000) 
        self.world = client.get_world()
        self.ego_vehicle = ego_vehicle
        self.carequipment = carequipment
        self.setup = env_setup

    def reset(self):
        self.destroy_actors()
        self.setup_env()

    def destroy_actors(self):
        destroy_list = ['*vehicle*', '*sensor*', '*walker*']
        for actor_type in destroy_list:
            for actor in self.world.get_actors().filter(actor_type):
                actor.destroy()


    def setup_env(self):
        self.setup()

    def step(self, action, **kwargs):
        carla_vehicle_control = action.vehicle_control
        self.ego_vehicle.apply_control(carla_vehicle_control)
        # compute reward 
        reward = 0 
        # termination flag
        done = False 
        

