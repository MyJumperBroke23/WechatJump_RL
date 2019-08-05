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


def test():
    model.eval()
    for episode in range(num_episodes):
        state = torch.Tensor(getLessProcessed()).unsqueeze(0)
        state_shape = state.shape
        state = state.view(state_shape[0], state_shape[3], state_shape[1], state_shape[2])
        action = model(state)
        os.system("adb shell input swipe 500 500 500 500 " + str(int(action)))
        time.sleep(1)


def trainJump(save_as, curr_checkpoint):
    model.train()
    for episode in range(num_episodes):
        variance = final_variance + (initial_variance - final_variance) * \
                        math.exp(-1. * episode / variance_decay)
        time.sleep(1.2)
        prev_score = getScore()
        if episode % 10 == 0:
            print("Score:", prev_score)
            print("-----------------------------------------")
            print("Episode:", episode)
        state = torch.Tensor(getLessProcessed()).unsqueeze(0)
        state_shape = state.shape
        state = state.view(state_shape[0], state_shape[3], state_shape[1], state_shape[2])
        mean = model(state)
        if episode % 10 == 0:
            print("Mean:", float(mean), "Deviation:", float(variance))
        m = Normal(mean, variance)
        action = m.sample()
        if episode % 10 == 0:
            print("Action:", action)
        os.system("adb shell input swipe 500 500 500 500 " + str(int(action)))
        time.sleep(0.5)
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

        if (episode + 1) % 1001 == 0:
            save = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            file_name = save_as + str((episode // 1000) + curr_checkpoint) + ".pth"
            torch.save(save, file_name)


def onDeath():
    time.sleep(1.9)
    if getScore() == -10:
        os.system("adb shell input tap 550 1700")
        time.sleep(0.5)
    else:
        image = getImage()
        thing = image[1000, 200] == [255, 255, 255]
        if (thing[0] == True and thing[1] == True and thing[2] == True):
            os.system("adb shell input tap 200 1400")
            time.sleep(0.1)
            os.system("adb shell input tap 550 1700")


save = torch.load("models/CNN_Policy_Scratch33.pth")
model = CNN_Policy(1)
model.load_state_dict(save["state_dict"], strict=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.load_state_dict(save['optimizer'])


trainJump("models/CNN_Policy_Scratch", 33)




'''
Jump:

state = torch.load("Jump7.pth")
model.load_state_dict(state["state_dict"])
optimizer.load_state_dict(state["optimizer"])
'''