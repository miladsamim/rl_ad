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
import replay_memory as rplm
import utils
from timeit import default_timer as timer 

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_FREQ=25
SAVE_FREQ=1000

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

    def __init__(self, environment, architecture, explore_rate, learning_rate,
                 batch_size, memory_capacity, num_episodes, learning_rate_drop_frame_limit,
                 target_update_frequency, discount=0.99, delta=1, model_name="torch_dqn_m", architecture_args=(4,5)):
        self.env = environment
        self.action_size = self.env.action_space_size
        self.avg_reward = None
        self.q_grid = None
        self.dqn = architecture(*architecture_args).to(DEVICE)
        self.target_dqn = architecture(*architecture_args).to(DEVICE)
        self.update_fixed_target_weights()
        self.learning_rate = 0.00025# learning_rate() # atari learning rate
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.explore_rate = explore_rate()
        self.criterion = nn.HuberLoss()

        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name if model_name else str(self.env)
        self.log_path = self.model_path + '/log'
        self.writer = SummaryWriter(self.log_path)

        # Training parameters setup
        self.target_update_frequency = target_update_frequency
        self.discount = discount
        self.replay_memory = rplm.Replay_Memory(memory_capacity, batch_size)
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
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.get_mini_batch(self.training_metadata)
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).to(DEVICE)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.int64).argmax(dim=1,keepdim=True).to(DEVICE)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(DEVICE)
        done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            self.dqn.eval() # don't use dropout when estimating targets
            self.target_dqn.eval()
            greedy_actions = self.dqn(next_state_batch).argmax(dim=1, keepdims=True)
            q_value_targets = reward_batch + self.discount * ((1 - done_batch) * self.target_dqn(next_state_batch))
            q_value_targets = q_value_targets.gather(1, greedy_actions)
        
        self.dqn.train()
        q_value = self.dqn(state_batch)
        q_value = q_value.gather(1, action_batch)

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
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                self.dqn.eval()
                return self.dqn(state).detach().cpu().argmax(dim=1).item()

    # Description: Calculates the reward at a given timestep
    # Parameters:
    # - reward: 		Number, the reward returned by the emulator
    # - in_grass: 		Boolean, gives wether car is in grass or not
    # - episode_frame: 	Integer, counting the number of frames elapsed in the given episode
    # Output: 			Number, the reward calculated
    def calculate_reward(self, reward, in_grass, episode_frame):
        if self.env.detect_grass and in_grass and episode_frame > 10:
            reward = -1
        return reward

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
            state, _ = self.env.reset()
            self.env.render()

            # Setting up parameters for the episode
            done = False
            epsilon = self.explore_rate.get(self.training_metadata)
            alpha = self.learning_rate
            print("Episode {0}/{1} \t Epsilon: {2:.2f} \t Alpha(lr): {3}".format(episode, self.training_metadata.num_episodes, epsilon, alpha))
            episode_frame = 0
            episode_reward =  0
            while not done:
                # Updating fixed target weights every #target_update_frequency frames
                if self.training_metadata.frame % self.target_update_frequency == 0 and (self.training_metadata.frame != 0):
                    self.update_fixed_target_weights()

                # Choosing and performing action and updating the replay memory
                action = self.get_action(state, epsilon)
                next_state, reward, done, info, in_grass = self.env.step(action)

                episode_reward += reward
                reward = self.calculate_reward(reward, in_grass, episode_frame)
                episode_frame += 1

                self.replay_memory.add(self, state, action, reward, next_state, done)

                # Performing experience replay if replay memory populated
                if self.replay_memory.length() > 10 * self.replay_memory.batch_size:
                    self.training_metadata.increment_frame()
                    self.experience_replay()
                state = next_state
                done = info['true_done']

            # Creating q_grid if not yet defined and calculating average q-value
            if self.replay_memory.length() > 100 * self.replay_memory.batch_size:
                self.q_grid = self.replay_memory.get_q_grid(size=128, training_metadata=self.training_metadata)
            avg_q = self.estimate_avg_q()

            # Saving tensorboard data and model weights
            if (episode % EVAL_FREQ == 0) and (episode != 0):
                score, std, rewards = self.test(num_test_episodes=5, visualize=True)
                print('{0} +- {1}'.format(score, std))
                self.writer.add_scalar('Test Reward (5 eps.)', score, episode / EVAL_FREQ)
                self.writer.add_scalar('Test Reward Std (5 eps.)', std, episode / EVAL_FREQ)
                if episode % SAVE_FREQ == 0:
                    self.save(self.model_path+f'_{episode}.pt')
                
            print(f'Epsiode {episode}/{self.training_metadata.num_episodes} | avg q value {avg_q} | Reward {episode_reward:.2f} | Frames {episode_frame}')

            self.writer.add_scalar('Avg. Q val', avg_q, episode)
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
            state, _ = self.env.reset(test=True)
            episode_reward = 0
            if not visualize:
                self.test_env.render()
            while not done:
                if visualize:
                    self.env.render()
                action = self.get_action(state, epsilon=0)
                next_state, reward, done, info, _ = self.env.step(action, test=True)
                state = next_state
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
            q_grid = torch.tensor(np.array(self.q_grid), dtype=torch.float32).to(DEVICE)
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
