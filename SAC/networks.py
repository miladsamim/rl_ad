import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# ---------- Constants ---------- # 

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# ---------- OpenAI core functions ---------- # 

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# ---------- Functions for network architecture block creation ---------- # 

def conv_block(c_input, c_output, *args, **kwargs):
    layers = []
    layers.append(nn.Conv2d(c_input,c_output, *args, **kwargs))
    layers.append(nn.BatchNorm2d(c_output))
    layers.append(nn.ReLU())
    return layers

def fc_block(c_input,c_output, *args, **kwargs):
    layers = []
    layers.append(nn.Linear(c_input,c_output, *args, **kwargs))
    layers.append(nn.ReLU())
    return layers

def cnn():
    layers = []
    layers.extend(conv_block(3,24,kernel_size=5, stride=2))
    layers.extend(conv_block(24,36,kernel_size=5, stride=2))
    layers.extend(conv_block(36,48,kernel_size=5, stride=2))
    layers.extend(conv_block(48,64,kernel_size=3, stride=1))
    layers.extend(fc_block(1152, 192))
    layers.extend(fc_block(192, 64))
    return nn.Sequential(*layers)


# ---------- Classes ---------- # 


class Actor(nn.Module):

    """
    CNN-LSTM Actor class.

    Input:
        - n_actions: The number of actions
        - h_size: The desired amout of neurons in each hidden layer
        - n_layers: Number of desired LSTM layers
        - action_limit: The limit for the actions

    """
    
    def __init__(self, n_actions, h_size, n_layers, action_limit):
        super().__init__()
        self.cnn = cnn()
        self.lstm = nn.LSTM(input_size = 64 + n_actions, hidden_size = h_size, num_layers=n_layers, batch_first=True)
        self.mu_layer = nn.Linear(h_size[-1], n_actions)
        self.log_std_layer = nn.Linear(h_size[-1], n_actions)
        self.action_limit = action_limit

    def forward(self, state, deterministic=False, with_logprob=True):
        cnn_out = self.cnn(state)

        _ , (h_output, _ ) = self.lstm(cnn_out)
        mu = self.mu_layer(h_output)
        log_std = self.log.std_layer(h_output)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        
        # Only used for evaluating policy at test time
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # Note: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.action_limit * pi_action

        return pi_action, logp_pi


class Critic(nn.Module):

    """
    CNN-LSTM Critic class.

    Input:
        - n_actions: The number of actions
        - h_size: The desired amout of neurons in each hidden layer
        - n_layers: Number of desired LSTM layers
    """
    
    def __init__(self, n_actions, h_size, n_layers):
        super().__init__()
        self.cnn = cnn()
        self.lstm = nn.LSTM(input_size = 64 + n_actions, hidden_size = h_size, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(h_size, 1)

    def forward(self, state, action):
        cnn_out = self.cnn(state)
        lstm_input = torch.cat([cnn_out, action], dim = -1)
        
        _ , (h_output, _ ) = self.lstm(lstm_input)
        
        q_values = self.out(h_output)

        return q_values


class ActorCritic(nn.Module):

    """
    Actor Critic Class that builds:
        - A policy network, pi
        - Two critic networks, q1 and q2
        - An act function that computes the computes an action using the policy network
    """

    def __init__(self, action_space, h_size = 256, n_layers = 4):
        super().__init__()
        n_actions = action_space.shape[0]
        action_limit = action_space.high[0]

        # Build policy (pi) and action-value functions (q1 and q2)
        self.pi = Actor(n_actions, h_size, n_layers, action_limit)
        self.q1 = Critic(n_actions, h_size, n_layers)
        self.q2 = Critic(n_actions, h_size, n_layers)

        def act(self, state, deterministic=False):
            with torch.no_grad():
                action, _ = self.pi(state, deterministic, False)
                return action.numpy()
