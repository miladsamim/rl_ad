import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
import random
import os
import subprocess
from parameters import MemoryBufferSimple
import utils
from collections import deque
from timeit import default_timer as timer 

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

EVAL_FREQ=25
SAVE_FREQ=100


#####################################  Description  ####################################################
# This file defines the class DQN_Agent. It uses Double DQN to train the neural network.
########################################################################################################

class DQN_Agent:


	# Description: Initializes the DQN_Agent object
	# Parameters:
	# - environment: 		Object supporting methods like 'reset', 'render', 'step' etc.
	# 			     		For an example see environment.py.
	# - architecture: 		Object supporting the method 'evaluate', usually a neural network. 
	# 				  		for exapmles see parameters/architectures.py.
	# - explore_rate: 		Object supporting the method 'get'. See parameters/explore_rates.py for examples.
	# - learning_rate: 		Object supporting the method 'get'. See parameters/learning_rates.py for examples.
	# - batch_size: 		Integer giving the size of the minibatch to be used in the optimization process.
	# - memory_capacity: 	Integer giving the size of the replay memory.
	# - num_episodes: 		Integer specifying the number of training episodes to be used. 
	# - learning_rate_drop_frame_limit:
	# 						Integer specifying by which frame during training the minimal explore rate should be reached 
	# - target_update_frequency:
	# 						Integer specifying the frequency at which the target network's weights are to be updated
	# - discount:			Number in (0,1) giving the discount factor
	# - delta:  			Number, the delta parameter in huber loss
	# - model_name:  		String, the name of the folder where the model is saved. 
	# Output: None

    def __init__(self, environment, architecture, architecture_args, explore_rate, learning_rate,
                 batch_size, memory_capacity, num_episodes, learning_rate_drop_frame_limit,
                 target_update_frequency, discount=0.99, delta=1, model_name=None):
        self.args = args = architecture_args
        self.env = environment
        self.action_size = self.env.action_space_size
        self.avg_reward = None
        self.q_grid = None
        self.dqn = architecture(args).to(args.device)
        self.target_dqn = architecture(args).to(args.device)
        self.update_fixed_target_weights()
        self.learning_rate = learning_rate#0.00025# learning_rate() # atari learning rate
        self.batch_size = batch_size
        parameters = [param for param in self.dqn.parameters() if param.requires_grad == True]
        self.optim = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.explore_rate = explore_rate()
        self.criterion = nn.HuberLoss()

        self.model_name = architecture.__name__
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + self.model_name
        self.log_path = self.model_path + '/log'
        self.writer = SummaryWriter(self.log_path)
        with open(self.model_path + '/' + self.model_name + '_args.txt', 'w') as f: 
            for key, value in architecture_args.items(): 
                f.write('%s:%s\n' % (key, value))


        # Training parameters setup
        self.target_update_frequency = target_update_frequency
        self.discount = discount
        self.replay_memory = MemoryBufferSimple(args.n_frames, memory_capacity)
        self.replay_memory_sampler = torch.utils.data.DataLoader(self.replay_memory, batch_size=batch_size, shuffle=True)
        # self.training_metadata = utils.Training_Metadata(frame=self.sess.run(self.frames), frame_limit=learning_rate_drop_frame_limit,
        # 												   episode=self.sess.run(self.episode), num_episodes=num_episodes)
        self.training_metadata = utils.Training_Metadata(frame=0, frame_limit=learning_rate_drop_frame_limit,
                                                         episode=0, num_episodes=num_episodes)
        self.delta = delta
        utils.document_parameters(self)
        self.start_time = timer()

    
    # Description: Performs one step of batch gradient descent on the DDQN loss function. 
    # Parameters:
    # - alpha: Number, the learning rate 
    # Output: None
    def experience_replay(self):
        args = self.args
        states, actions, rewards, dones = next(iter(self.replay_memory_sampler))
        X_img = states[0].transpose(0,1).to(args.device)
        X_sensor = torch.stack(states[1:-1], axis=0).permute(0,2,1).unsqueeze(-1).to(args.device)
        X_act = states[-1].transpose(0,1).to(args.device)
        cur_state = (X_img[:-1], X_sensor[:,:-1], X_act[:-1])
        next_state = (X_img[1:], X_sensor[:,1:], X_act[1:])
        actions = actions.unsqueeze(-1).to(args.device)
        rewards = rewards.unsqueeze(1).to(args.device)
        dones = dones.unsqueeze(1).to(args.device)

        with torch.no_grad():
            self.dqn.eval() # don't use dropout when estimating targets
            self.target_dqn.eval()
            greedy_actions = self.dqn(*next_state).argmax(dim=1, keepdims=True)
            q_value_targets = rewards + self.discount * ((1 - dones) * self.target_dqn(*next_state))
            q_value_targets = q_value_targets.gather(1, greedy_actions)
        
        self.dqn.train()
        q_value = self.dqn(*cur_state)
        q_value = q_value.gather(1, actions)

        loss = self.criterion(q_value, q_value_targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    # Description: Chooses action wrt an e-greedy policy. 
    # Parameters:
    # - state: 		Tensor representing a single state
    # - epsilon: 	Number in (0,1)
    # Output: 		Integer in the range 0...self.action_size-1 representing an action
    def get_action(self, state, epsilon):
        # Performing epsilon-greedy action selection
        if random.random() < epsilon:
            return self.env.sample_action_space()
        else:
            args = self.args
            state = self.replay_memory._process_states(state)
            X_img = state[0].unsqueeze(1).to(args.device)
            X_sensor = torch.stack(state[1:-1], axis=0)[...,None,None].to(args.device)
            X_act = state[-1].unsqueeze(1).to(args.device)
            state = (X_img, X_sensor, X_act)
            with torch.no_grad():
                self.dqn.eval()
                return self.dqn(*state).detach().cpu().argmax(dim=1).item()

    # Description: Updates the weights of the target network
    # Parameters: 	None
    # Output: 		None
    def update_fixed_target_weights(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    # Description: Trains the model
    # Parameters: 	None
    # Output: 		None
    def train(self):
        while self.training_metadata.episode < self.training_metadata.num_episodes:
            episode = self.training_metadata.episode
            self.training_metadata.increment_episode()

            # Setting up game environment
            state_img = self.env.reset()
            state = self.process_state(state_img, 0)
            state_frame_stack = deque(maxlen=self.args.n_frames)
            for i in range(self.args.n_frames):
                state_frame_stack.append(state)
            self.env.render()

            # Setting up parameters for the episode
            done = False
            epsilon = self.explore_rate.get(self.training_metadata)
            alpha = self.learning_rate
            print(self.model_name + ": Episode {0}/{1} \t Epsilon: {2:.2f} \t Alpha(lr): {3}".format(episode, self.training_metadata.num_episodes, epsilon, alpha))
            episode_frame = 0
            episode_reward =  0
            while not done:
                # Updating fixed target weights every #target_update_frequency frames
                if self.training_metadata.frame % self.target_update_frequency == 0 and (self.training_metadata.frame != 0):
                    self.update_fixed_target_weights()

                # Choosing and performing action and updating the replay memory
                action = self.get_action(state_frame_stack, epsilon)
                next_state_img, reward, done, info = self.env.step(action)
                next_state = self.process_state(next_state_img, action)

                episode_reward += reward
                episode_frame += 1

                self.replay_memory.add_experience(state, action, reward, done)

                # Performing experience replay if replay memory populated
                if self.replay_memory.__len__() > 10 * self.replay_memory.batch_size:
                    self.training_metadata.increment_frame()
                    self.experience_replay()
                # state = next_state
                state = next_state
                state_frame_stack.append(state)
                done = info['true_done']

            # Saving tensorboard data and model weights
            if (episode % EVAL_FREQ == 0) and (episode != 0):
                score, std, rewards = self.test(num_test_episodes=5, visualize=True)
                print('{0} +- {1}'.format(score, std))
                self.writer.add_scalar('Test Reward (5 eps.)', score, episode / EVAL_FREQ)
                self.writer.add_scalar('Test Reward Std (5 eps.)', std, episode / EVAL_FREQ)
                if episode % SAVE_FREQ == 0:
                    self.save(self.model_path + '/' + self.model_name + '.pt')
                
            print(f'Epsiode {episode}/{self.training_metadata.num_episodes} | Reward {episode_reward:.2f} | Frames {episode_frame}')

            self.writer.add_scalar('epsilon', epsilon, episode)
            self.writer.add_scalar('lr', alpha, episode)
            self.writer.add_scalar('Reward', episode_reward, episode)
            self.writer.add_scalar('Episode steps', episode_frame, episode)
            if not self.avg_reward:
                self.avg_reward = episode_reward
            else: 
                self.avg_reward = self.avg_reward  + (episode_reward - self.avg_reward)/episode
            self.writer.add_scalar('Avg. Reward', self.avg_reward, episode)
            episode_end_time = timer() - self.start_time 
            self.writer.add_scalar('Time', episode_end_time, episode)

    # Description: Tests the model
    # Parameters:
    # - num_test_episodes: 	Integer, giving the number of episodes to be tested over
    # - visualize: 			Boolean, gives whether should render the testing gameplay
    def test(self, num_test_episodes, visualize):
        rewards = []
        for episode in range(num_test_episodes):
            done = False
            state_img = self.env.reset(test=True)
            state = self.process_state(state_img, 0)
            state_frame_stack = deque(maxlen=self.args.n_frames)
            for i in range(self.args.n_frames):
                state_frame_stack.append(state)
            episode_reward = 0
            if not visualize:
                self.test_env.render()
            while not done:
                if visualize:
                    self.env.render()
                action = self.get_action(state_frame_stack, epsilon=0)
                next_state_img, reward, done, info = self.env.step(action, test=True)
                next_state = self.process_state(next_state_img, action)
                state = next_state
                state_frame_stack.append(state)
                episode_reward += reward
                done = info['true_done']
            rewards.append(episode_reward)
        return np.mean(rewards), np.std(rewards), rewards

    # Description: Returns average Q-value over some number of fixed tracks
    # Parameters: 	None
    # Output: 		None
    def estimate_avg_q(self):
        if not self.q_grid:
            return 0
        else: 
            q_grid = torch.tensor(np.array(self.q_grid), dtype=torch.float32).to(self.args.device)
            with torch.no_grad():
                self.dqn.eval()
                q_values = self.dqn(q_grid).amax(dim=1)
            return q_values.mean().detach().cpu().item()
                        
    # Description: Loads a model trained in a previous session
    # Parameters:
    # - path: 	String, giving the path to the checkpoint file to be loaded
    # Output:	None
    def load(self, path):
        self.dqn.load_state_dict(torch.load(path))
        self.update_fixed_target_weights()
    
    def save(self, path):
        torch.save(self.dqn.state_dict(), path)

    def process_state(self, state_img, action_idx):
        """Process state so it can be used with transformer model"""
        state_img = state_img/255.0 
        data_board = state_img[84:, 12:]
        steering, speed, gyro, abs1, abs2, abs3, abs4 = self.compute_steering_speed_gyro_abs(data_board)
        state = {'img': state_img,
                'steering': steering,
                'speed': speed,
                'gyro': gyro,
                'abs1': abs1,
                'abs2': abs2,
                'abs3': abs3,
                'abs4': abs4,
                'action_idx': action_idx}
        return state


    def compute_steering_speed_gyro_abs(self, data_board):
        "process data board to extract sensor measurements"
        right_steering = data_board[6, 36:46].mean()/255
        left_steering = data_board[6, 26:36].mean()/255
        steering = (right_steering - left_steering + 1.0)/2
        
        left_gyro = data_board[6, 46:60].mean()/255
        right_gyro = data_board[6, 60:76].mean()/255
        gyro = (right_gyro - left_gyro + 1.0)/2
        
        speed = data_board[:, 0][:-2].mean()/255
        abs1 = data_board[:, 6][:-2].mean()/255
        abs2 = data_board[:, 8][:-2].mean()/255
        abs3 = data_board[:, 10][:-2].mean()/255
        abs4 = data_board[:, 12][:-2].mean()/255
        
        return [steering, speed, gyro, abs1, abs2, abs3, abs4]
