import os 
import sys
sys.dont_write_bytecode = True
os.chdir(sys.path[0])

import gym
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
import random
from parameters import MemoryBufferSeparated
from parameters.pact import PACTPretrain
import utils
from collections import deque
from timeit import default_timer as timer 
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

EVAL_FREQ=25
SAVE_FREQ=100
from importlib.metadata import version
USE_V2 = '0.21.0' < version('gym')


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
                 target_update_frequency, discount=0.99, process_state=True, use_all_timesteps=False,
                 delta=1, model_name=None):
        self.args = args = architecture_args
        self.env = environment
        self.action_size = self.env.action_space_size
        self.avg_reward = None
        self.q_grid = None
        self.dqn = architecture(args).to(args.device)
        self.target_dqn = architecture(args).to(args.device)
        self.pact_model = PACTPretrain(args).to(args.device)
        self.update_fixed_target_weights()
        self.learning_rate = learning_rate#0.00025# learning_rate() # atari learning rate
        self.batch_size = batch_size
        self.dqn_optim = torch.optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.pact_optim = torch.optim.Adam(self.pact_model.parameters())
        self.explore_rate = explore_rate()
        self.dqn_criterion = nn.HuberLoss()
        self.pact_criterion = nn.MSELoss()

        self.model_name = architecture.__name__ + f'_{self.args.n_frames}f_' + f'{self.args.residual}res'
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + self.model_name
        self.log_path = self.model_path + '/log'
        self.writer = SummaryWriter(self.log_path)
        self.stats_file_path = self.model_path + '/stats.csv'
        self.stats_buffer = []
        with open(self.stats_file_path, 'w') as f:
            f.write('Episode,NumOfEpisodes,Epsilon,lr,Reward,Frames\n')
        with open(self.model_path + '/' + self.model_name + '_args.txt', 'w') as f: 
            for key, value in architecture_args.items(): 
                f.write('%s:%s\n' % (key, value))

        # Training parameters setup
        self.target_update_frequency = target_update_frequency
        self.discount = discount
        # self.replay_memory = MemoryBufferSimple(args.n_frames, memory_capacity)
        self.process = process_state
        self.use_all_timesteps = use_all_timesteps
        self.replay_memory = MemoryBufferSeparated(args.n_frames, memory_capacity, process_state=process_state)
        self.replay_memory_sampler = torch.utils.data.DataLoader(self.replay_memory, batch_size=batch_size, shuffle=True)
        # self.training_metadata = utils.Training_Metadata(frame=self.sess.run(self.frames), frame_limit=learning_rate_drop_frame_limit,
        # 												   episode=self.sess.run(self.episode), num_episodes=num_episodes)
        self.training_metadata = utils.Training_Metadata(frame=0, frame_limit=learning_rate_drop_frame_limit,
                                                         episode=0, num_episodes=num_episodes)
        self.delta = delta
        utils.document_parameters(self)
        self.start_time = timer()
        
        self.act_idx_2_real = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, 1, 0], 
                                            [0, 0, 0.8], [0, 0, 0]], dtype=torch.float32, device=args.device)

    
    # Description: Performs one step of batch gradient descent on the DDQN loss function. 
    # Parameters:
    # - alpha: Number, the learning rate 
    # Output: None
    def experience_replay(self):
        args = self.args
        states, actions, rewards, dones = next(iter(self.replay_memory_sampler))
        X_img = states[0].transpose(0,1).to(args.device)
        if self.process_state:
            X_sensor = torch.stack(states[1:-1], axis=0).permute(0,2,1,3).to(args.device)
        else: # adj for framestacking
            X_sensor = torch.stack(states[1:-1], axis=0).unsqueeze(-1).permute(0,2,1,3).to(args.device)
        X_act = states[-1].transpose(0,1).to(args.device)
        cur_state = (X_img[:-1], X_sensor[:,:-1], X_act[:-1])
        next_state = (X_img[1:], X_sensor[:, 1:], X_act[1:])
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

        loss = self.dqn_criterion(q_value, q_value_targets)

        self.dqn_optim.zero_grad()
        loss.backward()
        self.dqn_optim.step()

    def experience_replay_all_steps(self):
        args = self.args
        states, actions, rewards, dones = next(iter(self.replay_memory_sampler))
        X_img = states[0].transpose(0,1).to(args.device)
        X_sensor = torch.stack(states[1:-1], axis=0).permute(0,2,1,3).to(args.device)
        X_act_idx = states[-1].transpose(0,1).to(args.device)
        X_act = self.act_idx_2_real[X_act_idx].permute(2,0,1).unsqueeze(-1) # map discrete actions to real vals
        batch = {'state': {'X_img': X_img, 'X_sensor': X_sensor},
                 'action': {'X_act_idx': X_act_idx, 'X_act': X_act}}

        # Pretrain step 
        self.pact_model.train()
        state_preds, state_targets = self.pact_model(batch)
        pretrain_loss = self.pact_criterion(state_preds[:-1], state_targets[1:]) # predict t+1
        
        # DQN experience replay for every t
        cur_state = state_targets[:-1]
        next_state = state_targets[1:]
        actions = X_act_idx[:-1].unsqueeze(-1) 
        rewards = rewards[None,...,None].to(args.device)
        dones = dones[None,...,None].to(args.device)

        with torch.no_grad():
            self.dqn.eval() # don't use dropout when estimating targets
            self.target_dqn.eval()
            greedy_actions = self.dqn(next_state).argmax(dim=2, keepdims=True)
            q_value_targets = rewards + self.discount * ((1 - dones) * self.target_dqn(next_state))
            q_value_targets = q_value_targets.gather(2, greedy_actions)

        self.dqn.train()
        q_value = self.dqn(cur_state)
        q_value = q_value.gather(2, actions)

        dqn_loss = self.dqn_criterion(q_value, q_value_targets)

        # Gradient step
        self.pact_optim.zero_grad()
        self.dqn_optim.zero_grad()
        loss = dqn_loss + pretrain_loss
        loss.backward()
        self.pact_optim.step()
        self.dqn_optim.step()

        return pretrain_loss.detach().cpu().item()

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
            X_sensor = torch.stack(state[1:-1], axis=0)[...,None].to(args.device)
            X_act_idx = state[-1].unsqueeze(1).to(args.device)
            X_act = self.act_idx_2_real[X_act_idx].permute(2,0,1).unsqueeze(-1) # map discrete actions to real vals 
            state_dict = {'state': {'X_img': X_img, 'X_sensor': X_sensor},
                    'action': {'X_act_idx': X_act_idx, 'X_act': X_act}}
            with torch.no_grad():
                self.pact_model.eval()
                self.dqn.eval()
                state_preds, state_targets = self.pact_model(state_dict)
                return self.dqn(state_targets[-1]).detach().cpu().argmax(dim=1).item()

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
            state_img, _ = self.env.reset()
            state = self.process_state(state_img, 0, process=self.process)
            self.replay_memory.add_experience(state, 0, 0, False, new_episode=True) # initialize new ep in buffer 
            state_frame_stack = deque(maxlen=1)#deque(maxlen=self.args.n_frames)
            for i in range(self.args.n_frames):
                state_frame_stack.append(state)
            if not USE_V2:
                self.env.render()

            # Setting up parameters for the episode
            done = False
            epsilon = self.explore_rate.get(self.training_metadata)
            alpha = self.learning_rate
            print(self.model_name + ": Episode {0}/{1} \t Epsilon: {2:.2f} \t Alpha(lr): {3}".format(episode, self.training_metadata.num_episodes, epsilon, alpha))
            episode_frame = 0
            episode_reward = 0
            ep_pretrain_losses = []
            while not done:
                # Updating fixed target weights every #target_update_frequency frames
                if self.training_metadata.frame % self.target_update_frequency == 0 and (self.training_metadata.frame != 0):
                    self.update_fixed_target_weights()

                # Choosing and performing action and updating the replay memory
                action = self.get_action(state_frame_stack, epsilon)
                next_state_img, reward, done, info, in_grass = self.env.step(action)
                next_state = self.process_state(next_state_img, action, process=self.process)

                episode_reward += reward
                episode_frame += 1

                self.replay_memory.add_experience(state, action, reward, done, new_episode=False)
                # self.replay_memory.add_experience(state, action, reward, done)

                # Performing experience replay if replay memory populated
                if self.replay_memory.__len__() > 10 * self.replay_memory.batch_size:
                    self.training_metadata.increment_frame()
                    if self.use_all_timesteps:
                        pretrain_loss = self.experience_replay_all_steps()
                        ep_pretrain_losses.append(pretrain_loss)
                    else:
                        self.experience_replay()
                # state = next_state
                state = next_state
                state_frame_stack.append(state)
                done = info['true_done']
            
            ep_pretrain_loss = np.mean(ep_pretrain_losses)
            print(f'Epsiode {episode}/{self.training_metadata.num_episodes} | Reward {episode_reward:.2f} | Frames {episode_frame} | PretrainLoss {ep_pretrain_loss:.2f}')
            self.stats_buffer.append(f'{episode}, {self.training_metadata.num_episodes}, {epsilon}, {alpha}, {episode_reward:.2f}, {episode_frame}\n')

            # Saving tensorboard data and model weights
            if (episode % EVAL_FREQ == 0) and (episode != 0):
                score, std, rewards = self.test(num_test_episodes=5, visualize=True)
                print('{0} +- {1}'.format(score, std))
                self.writer.add_scalar('Test Reward (5 eps.)', score, episode / EVAL_FREQ)
                self.writer.add_scalar('Test Reward Std (5 eps.)', std, episode / EVAL_FREQ)
                with open(self.stats_file_path, 'a') as fp:
                    fp.writelines(self.stats_buffer)
                    self.stats_buffer = []
            if episode % SAVE_FREQ == 0:
                self.save(self.model_path + '/' + self.model_name + f'_{episode}.pt')
                # os.popen('sh push.sh')    

            self.writer.add_scalar('epsilon', epsilon, episode)
            self.writer.add_scalar('lr', alpha, episode)
            self.writer.add_scalar('Reward', episode_reward, episode)
            self.writer.add_scalar('Episode steps', episode_frame, episode)
            self.writer.add_scalar('Episode Pretrain Loss', ep_pretrain_loss, episode)
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
            state_img, _ = self.env.reset(test=True)
            state = self.process_state(state_img, 0, process=self.process)
            state_frame_stack = deque(maxlen=1)#deque(maxlen=self.args.n_frames)
            for i in range(self.args.n_frames):
                state_frame_stack.append(state)
            episode_reward = 0
            if not visualize and not USE_V2:
                self.test_env.render()
            while not done:
                if visualize and not USE_V2:
                    self.env.render()
                action = self.get_action(state_frame_stack, epsilon=0)
                next_state_img, reward, done, info, in_grass = self.env.step(action, test=True)
                next_state = self.process_state(next_state_img, action, process=self.process)
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

    def process_state(self, state_img, action_idx, process):
        """Process state so it can be used with transformer model"""
        if process:
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
        else: 
            state = {'img': state_img,
                    'steering': 0,
                    'speed': 0,
                    'gyro': 0,
                    'abs1': 0,
                    'abs2': 0,
                    'abs3': 0,
                    'abs4': 0,
                    'action_idx': 0}
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
