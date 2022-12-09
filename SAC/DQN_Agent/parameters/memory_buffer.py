import torch 
import itertools
import numpy as np
from collections import deque 

class MemoryBufferSimple(torch.utils.data.Dataset):
    """Assumme that episodes are long, so probability that we will land in a part which
       starts a new episode is small, so just store in simple arrays"""
    def __init__(self, num_frames, max_buffer_sz=25_000, batch_size=32):
        self.memory_capacity = max_buffer_sz
        self.batch_size = batch_size
        self.states = deque(maxlen=max_buffer_sz)
        self.actions = deque(maxlen=max_buffer_sz)
        self.rewards = deque(maxlen=max_buffer_sz)
        self.dones = deque(maxlen=max_buffer_sz)
        self.active_ep_idx = -1 
        self.num_frames = num_frames
    
    def __len__(self):
        length = len(self.states) - self.num_frames - 1 # -1 as we need s_t+1 also
        if length < 0:
            return 1 # to avoid throwing error, when defining randomsampler before populating data
        else:
            return length

    def add_experience(self, state:dict, action:list, reward:float, done:bool):
        """states are at t+1, so when get_item we have to select actions, rewards one forward
           state>dict: which can be processed by _process_states
           action>int: which maps to action value in action_space list 
           reward>float: float value representing reward r(s'|s,a)
           done>bool: whether the episode is terminated at current step. Influences the target update"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def _np_img_to_tensor(self, img):
        img = np.array(img) # num_frames+1 x W x H x C
        return torch.tensor(img.transpose(2,0,1), dtype=torch.float32)

    def _process_states(self, states):
        """This function should be passed into the class as it could depend on the
           sensor setup. This configuration works for the HDSensor setup."""
        img_x = []; steering_x = []; speed_x = []; gyro_x = []; abs1_x = [];
        abs2_x = []; abs3_x = []; abs4_x = []; act_x = []; 
        for state in states:
            img_x.append(self._np_img_to_tensor(state['img']))
            steering_x.append(torch.tensor(state['steering'], dtype=torch.float32))
            speed_x.append(torch.tensor(state['speed'], dtype=torch.float32))
            gyro_x.append(torch.tensor(state['gyro'], dtype=torch.float32))
            abs1_x.append(torch.tensor(state['abs1'], dtype=torch.float32))
            abs2_x.append(torch.tensor(state['abs2'], dtype=torch.float32))
            abs3_x.append(torch.tensor(state['abs3'], dtype=torch.float32))
            abs4_x.append(torch.tensor(state['abs4'], dtype=torch.float32))
            act_x.append(torch.tensor(state['action_idx'], dtype=torch.long))

        img_x = torch.stack(img_x)
        img_x = img_x.transpose(1,2)
        # print(img_x.shape)
        steering_x = torch.stack(steering_x)
        speed_x = torch.stack(speed_x)
        gyro_x = torch.stack(gyro_x)
        abs1_x = torch.stack(abs1_x)
        abs2_x = torch.stack(abs2_x)
        abs3_x = torch.stack(abs3_x)
        abs4_x = torch.stack(abs4_x)
        act_x = torch.stack(act_x)
        return [img_x, steering_x, speed_x, gyro_x, abs1_x, abs2_x, abs3_x, abs4_x, act_x]
        
    def __getitem__(self, idx):
        end_idx = idx + self.num_frames + 1 # +1 as we need next state as well
        states = self._process_states(itertools.islice(self.states, idx, end_idx))
        # end_idx - 1 as it is adjusted in the training loop
        action = torch.tensor(self.actions[end_idx-1], dtype=torch.int64)
        reward = torch.tensor(self.rewards[end_idx-1], dtype=torch.float32)
        dones = torch.tensor(self.dones[end_idx-1], dtype=torch.float32)
        return states, action, reward, dones