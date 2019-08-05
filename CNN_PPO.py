import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import math
import time
import os
from scoreCheck import getReward, getScore
from preprocess import getLessProcessed, getImage
from CNN_DQN import onDeath

class CNN_Policy(nn.Module):
    def __init__(self, output_dim):
        super(CNN_Policy, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.linear = nn.Linear(950400, 1)

        self.linear.bias.data[0] = 500

    def forward(self, x): # Outputs mean of Gaussian distribution
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.linear(x))


class Memory:  # Stores actions, states, probs, and rewards
    def __init__(self):
        self.action = []
        self.state = []
        self.logprob = []
        self.reward = []

    def add(self, action, state, logprob, reward):
        self.action.append(action)
        self.state.append(state)
        self.logprob.append(logprob)
        self.reward.append(reward)

    def clear_mem(self):
        self.action.clear()
        self.state.clear()
        self.logprob.clear()
        self.reward.clear()

    def return_mem(self):
        return self.action, self.state, self.logprob, self.reward


class PPO:
    def __init__(self, output_dim, discount, clip_factor, learning_rate):
        self.memory = Memory()
        self.model = CNN_Policy(output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.discount = discount
        self.clip_factor = clip_factor

    def a2c(self, state): # Returns action mean
        return self.model(state)

    def add_mem(self, action, state, logprob, reward):
        self.memory.add(action, state, logprob, reward)

    def optimize_model(self, epochs, variance):
        self.model.train()
        actions, states, old_probs, rewards = self.memory.return_mem()
        old_probs = torch.Tensor(old_probs).detach()
        for epoch in range(epochs):
            for i in range(len(states)):
                new_action_mean = self.model(states[i])
                dist = Normal(new_action_mean, variance)
                dist_entropy = dist.entropy()
                new_prob = dist.log_prob(actions[i])

                r = torch.exp(new_prob - old_probs[i])

                # Loss follows the PPO loss of the (new prob of action / old prob of action) * reward clamped between 2 values
                actor_loss = -min(r * rewards[i],
                                  torch.clamp(r, 1-self.clip_factor, 1+self.clip_factor) * rewards[i])

                actor_loss = actor_loss - (0.01 * dist_entropy) # Small bonus for entropy

                self.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()
        self.memory.clear_mem()

    def save_agent(self, name):
        save = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(save, name)

    def load_agent(self, load):
        save = torch.load(load)
        self.model.load_state_dict(save["state_dict"])
        self.optimizer.load_state_dict(save["optimizer"])


disc = 0.99
clip_f = 0.2
lr = 3e-4
update = 10
k = 3
num_episodes = 10000

Agent = PPO(output_dim=1, discount=disc, clip_factor=clip_f, learning_rate=lr)
Agent.load_agent("models/PPO_6.pth")

initial_variance = 20
final_variance = 1
variance_decay = 5000


def trainJump(save, save_as=None, curr_checkpoint=None):
    Agent.model.eval()
    for episode in range(num_episodes):
        time.sleep(1.2)
        prev_score = getScore()

        if episode % 10 == 0:
            print("Score:", prev_score)
            print("-----------------------------------------")
            print("Episode:", episode)

        # Get state
        state = torch.Tensor(getLessProcessed()).unsqueeze(0)
        state_shape = state.shape
        state = state.view(state_shape[0], state_shape[3], state_shape[1], state_shape[2])

        # Calculate mean and variance
        mean = Agent.a2c(state)
        variance = final_variance + (initial_variance - final_variance) * \
                        math.exp(-1. * episode / variance_decay)

        # Construct normal distribution and sample action
        if episode % 10 == 0:
            print("Mean:", float(mean), "Deviation:", float(variance))
        m = Normal(mean, variance)
        action = m.sample()
        log_prob = m.log_prob(action)
        if episode % 10 == 0:
            print("Action:", action)
        os.system("adb shell input swipe 500 500 500 500 " + str(int(action)))

        # Get reward and push action, state, log_prob, and reward to action
        time.sleep(0.5)
        reward = getReward(prev_score)
        if reward >= 2:
            reward = 10
        elif reward == 1:
            reward = 0.1
        elif reward < 0:
            reward = -10
            onDeath()
        Agent.add_mem(action,state,log_prob,reward)
        if episode % 10 == 0:
            print("Reward:", reward)
        if episode % 10 == 0:
            print("-----------------------------------------")

        # Optimize model
        if (episode + 1) % (k+1) == 0:
            Agent.optimize_model(k, variance)

        if save:
            if (episode + 1) % 1001 == 0:
                Agent.save_agent(save_as + str(curr_checkpoint + (episode // 1000)) + ".pth")


