import random
import numpy as np
from collections import deque
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class CarRacingDQNAgent:
    def __init__(
        self,
        action_space    = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ],
        frame_stack_num = 3,
        memory_size     = 5000,
        gamma           = 0.95,  # discount rate
        epsilon         = 1.0,   # exploration rate
        epsilon_min     = 0.1,
        epsilon_decay   = 0.9999,
        learning_rate   = 0.001,
        device = 'cuda'
    ):
        self.action_space    = action_space
        self.frame_stack_num = frame_stack_num
        self.memory          = deque(maxlen=memory_size)
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        self.device = device 
        self.model           = self.build_model()
        self.target_model    = self.build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 12, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(432, 216),
            nn.ReLU(),
            nn.Linear(216, len(self.action_space))
        )
        return model.to(self.device)

    def update_target_model(self):
        with torch.no_grad():
            for target_weights, model_weights in zip(self.target_model.parameters(),self.model.parameters()):
                target_weights.copy_(model_weights.data)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            act_values = self.model(state).detach().cpu().numpy()
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def replay(self, batch_size):
        self.model.train()
        self.optimizer.zero_grad()
        minibatch = random.sample(self.memory, batch_size)
        cur_q_stack = []
        train_target_stack = []
        for state, action_index, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            cur_q = self.model(torch.unsqueeze(state, axis=0))[0][action_index]
            # target = cur_q.detach().clone()
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    t = self.target_model(torch.unsqueeze(next_state, axis=0))[0]
                    target = reward + self.gamma * torch.argmax(t)
            cur_q_stack.append(cur_q)
            train_target_stack.append(target)

        cur_q_stack = torch.vstack(cur_q_stack)
        train_target_stack = torch.vstack(train_target_stack)

 
        loss = self.criterion(cur_q_stack, train_target_stack)
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay2(self, batch_size):
        self.model.train()
        states, actions_indicies, rewards, next_states, dones = zip(*random.sample(self.memory, batch_size))
        states = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_indicies = torch.tensor(np.asarray(actions_indicies), dtype=torch.long, device=self.device).reshape(-1,1)
        rewards = torch.tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.asarray(dones), dtype=torch.long, device=self.device)

        with torch.no_grad():
            # Compute the next Q-values using the target network
            next_q_values = self.target_model(next_states)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)
            # 1-step TD target
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Get current Q-values estimates
        current_q_values = self.model(states)

        # Retrieve the q-values for the actions from the replay buffer
        current_q_values = torch.gather(current_q_values, dim=1, index=actions_indicies)

        # Compute Huber loss (less sensitive to outliers)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
 
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.update_target_model()

    def save(self, name):
        torch.save(self.target_model.state_dict(), name)
