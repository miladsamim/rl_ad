import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from PIL import Image
import utils
import time

from importlib.metadata import version
USE_V2 = '0.21.0' < version('gym') 

class CarRacing:

    # Parameters
    # - type: Name of environment. Default is classic Car Racing game, but can be changed to introduce perturbations in environment
    # - history_pick: Size of history
    # - seed: List of seeds to sample from during training. Default is none (random games)
    def __init__(self, type="CarRacing", history_pick=4, seed=None, detect_edges=False, detect_grass=False, flip=False,
                 process_state=True, use_frame_skip=True, use_episode_flipping=True, render=False):
        self.name = type + str(time.time())
        if USE_V2:
            if render:
                self.env = gym.make(type + '-v2', new_step_api=True, render_mode='human')
            else:
                self.env = gym.make(type + '-v2', new_step_api=True)
        else:
            self.env = gym.make(type + '-v0')
        self.image_dimension = [96,96]
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.image_dimension)
        self.action_space_size = 5
        self.state_shape = [None, self.history_pick] + list(self.image_dimension)
        self.history = []
        self.action_dict = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0.8], 4: [0, 0, 0]}
        self.seed = seed
        self.detect_edges = detect_edges
        self.detect_grass = detect_grass
        self.flip = flip
        self.flip_episode = False
        self.process_state = process_state
        self.use_frame_skip = use_frame_skip
        self.use_episode_flipping = use_episode_flipping

    # returns a random action
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        if self.flip_episode and action <= 1:
            action = 1 - action
        return self.action_dict[action]

    # resets the environment and returns the initial state
    def reset(self, test=False):
        if self.use_episode_flipping:
            self.flip_episode = random.random() > 0.5 and not test and self.flip
        else:
            self.flip_episode = False 
        
        # discern between gym versions
        if self.seed:
            state = self.env.reset(seed=random.choice(self.seed))
        else:
            state = self.env.reset()

        if self.process_state:
            return self.process(state)
        else: 
            return state, None

    # take action 
    def step(self, action, test=False):
        action = self.map_action(action)
        total_reward = 0
        n = 1 if test or (not self.use_frame_skip) else random.choice([2, 3, 4])
        for i in range(n):
            if USE_V2:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            else:
                next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            info = {'true_done': done}
            if done: break
        if self.process_state:
            processed_next_state, in_grass = self.process(next_state)    
            return processed_next_state, total_reward, done, info, in_grass
        else:
            return next_state, total_reward, done, info, None 

    def render(self):
        self.env.render()

    # process state and return the current history
    def process(self, state):
        self.add_history(state)
        in_grass = utils.in_grass(state) 
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.image_dimension)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result, in_grass

    def add_history(self, state):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        temp = utils.process_image(state, detect_edges=self.detect_edges, flip=self.flip_episode)
        self.history.append(temp)

    def __str__(self):
    	return self.name + '\nseed: {0}\nactions: {1}'.format(self.seed, self.action_dict)