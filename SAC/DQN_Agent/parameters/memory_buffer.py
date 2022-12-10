import torch 
import itertools
import numpy as np

import random
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

class MemoryBufferSeparated(torch.utils.data.Dataset):
    """Assumme that each episode has at least num_frames+1 experieneces.
       Assumme that max length of any episode is smaller than max_buffer_sz.
       else it will be discarded. When max is reached oldest episodes will be fully dropped until under max again."""
    def __init__(self, num_frames, max_buffer_sz=25_000, batch_size=32):
        self.memory_capacity = max_buffer_sz
        self.batch_size = batch_size
        self.avg_ep_len = None 
        self.ep_lengths = deque()
        self.ep_states = deque()
        self.ep_actions = deque()
        self.ep_rewards = deque()
        self.ep_dones = deque()
        self.active_ep_idx = -1 
        self.num_frames = num_frames
        self.ep_min_len = 10 # min 1 is required here
    
    def __len__(self):
        n_states = sum(self.ep_lengths)
        n_eps = len(self.ep_lengths)
        length = n_states - (self.num_frames+1) * n_eps # +1 as we need to extract s_t+1 also
        if length <= 0:
            return 1 # to avoid issue with defining randomsampler before populating buffer
        else:
            return length
        # return len(self.states) - self.num_frames#sum(self.ep_lengths)

    def add_experience(self, state, action, reward, done, new_episode=False):
        """states are at t+1, so when get_item we have to select actions, rewards one forward"""
        if new_episode:
            # detect issues with previous epiosde
            if len(self.ep_lengths) > 0 and self.ep_lengths[-1] < self.num_frames + self.ep_min_len: # too small, ep might have crashed
                # pop previous
                self.ep_lengths.pop()
                self.ep_states.pop()
                self.ep_actions.pop()
                self.ep_rewards.pop()
                self.ep_dones.pop()
            self.ep_lengths.append(0)
            self.ep_states.append([])
            self.ep_actions.append([])
            self.ep_rewards.append([])
            self.ep_dones.append([])

        self.ep_lengths[-1] += 1 
        if self.memory_capacity < sum(self.ep_lengths):
            self.flush_experiences()

        self.ep_states[-1].append(state)
        self.ep_actions[-1].append(action)
        self.ep_rewards[-1].append(reward)
        self.ep_dones[-1].append(done)

    def flush_experiences(self):
        buffer_sz = sum(self.ep_lengths)
        to_flush = buffer_sz - self.memory_capacity
        flushed = 0
        while flushed < to_flush:
            flushed += self.ep_lengths.popleft()
            self.ep_states.popleft()
            self.ep_actions.popleft()
            self.ep_rewards.popleft()
            self.ep_dones.popleft()
        
    def _np_img_to_tensor(self, img):
        img = np.array(img) # num_frames+1 x W x H x C
        return torch.tensor(np.expand_dims(img.transpose(0,3,1,2), axis=0), dtype=torch.float32)

    def _process_states(self, states):
        """This function should be passed into the class as it could depend on the
           sensor setup. This configuration works for the HDSensor setup."""
        img_x, steering_x, speed_x, gyro_x, abs1_x, abs2_x, abs3_x, abs4_x, act_x = zip(*[state.values() for state in states])
        img_x = torch.tensor(np.stack(img_x).transpose(0,3,1,2), dtype=torch.float32)
        img_x = img_x.transpose(1,2) # due to using default carracer processing
        steering_x = torch.tensor(steering_x, dtype=torch.float32).unsqueeze(-1)
        speed_x = torch.tensor(speed_x, dtype=torch.float32).unsqueeze(-1)
        gyro_x = torch.tensor(gyro_x, dtype=torch.float32).unsqueeze(-1)
        abs1_x = torch.tensor(abs1_x, dtype=torch.float32).unsqueeze(-1)
        abs2_x = torch.tensor(abs2_x, dtype=torch.float32).unsqueeze(-1)
        abs3_x = torch.tensor(abs3_x, dtype=torch.float32).unsqueeze(-1)
        abs4_x = torch.tensor(abs4_x, dtype=torch.float32).unsqueeze(-1)
        act_x = torch.tensor(act_x, dtype=torch.float32).unsqueeze(-1)

        return [img_x, steering_x, speed_x, gyro_x, abs1_x, abs2_x, abs3_x, abs4_x, act_x]
        
    def __getitem__(self, idx):
        ep_len = 0 
        while ep_len < self.num_frames + self.ep_min_len:
            ep_idx = random.choice(range(len(self.ep_lengths)))
            ep_len = self.ep_lengths[ep_idx]
        t_idx = random.choice(range(self.ep_lengths[ep_idx]-self.num_frames-1)) # -1 as we need s_t+1 also
        t_end_idx = t_idx + self.num_frames + 1 # +1 as we need next state as well
        states = self._process_states(self.ep_states[ep_idx][t_idx:t_end_idx])
        # t_end_idx - 1 as it is adjusted in the training loop
        action = torch.tensor(self.ep_actions[ep_idx][t_end_idx-1], dtype=torch.int64) 
        reward = torch.tensor(self.ep_rewards[ep_idx][t_end_idx-1], dtype=torch.float32)
        dones = torch.tensor(self.ep_rewards[ep_idx][t_end_idx-1], dtype=torch.float32)
        return states, action, reward, dones