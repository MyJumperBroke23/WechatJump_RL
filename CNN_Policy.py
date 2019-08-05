import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import math
import numpy as np
import os
from torch.distributions import Normal
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
        self.linear = nn.Linear(950400, output_dim)

    def forward(self, x): # Outputs mean of Gaussian distribution
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.linear(x))


num_episodes = 9000

test_length = 100

initial_variance = 40
final_variance = 1
variance_decay = 5000
model = CNN_Policy(1)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def trainJump(save, save_as=None, curr_checkpoint=None):
    model.train()
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

        # Construct distribution based on calculated mean and variance
        mean = model(state)
        variance = final_variance + (initial_variance - final_variance) * \
                        math.exp(-1. * episode / variance_decay)
        if episode % 10 == 0:
            print("Mean:", float(mean), "Deviation:", float(variance))
        m = Normal(mean, variance)

        # Sample and perform action
        action = m.sample()
        if episode % 10 == 0:
            print("Action:", action)
        os.system("adb shell input swipe 500 500 500 500 " + str(int(action)))
        time.sleep(0.5)

        # Get reward and optimize model
        reward = getReward(prev_score)
        if reward >= 2:
            reward = 10
        elif reward == 1:
            reward = 0.3
        elif reward < 0:
            reward = -10
            onDeath()
        if episode % 1 == 0:
            print("Reward:", reward)
        loss = -m.log_prob(action) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print("-----------------------------------------")

        if save:
            if (episode + 1) % 1001 == 0:
                save_file = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                file_name = save_as + str((episode // 1000) + curr_checkpoint) + ".pth"
                torch.save(save_file, file_name)


