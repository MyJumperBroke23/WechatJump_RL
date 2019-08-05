import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import time
import numpy as np
import os
from torch.distributions import Normal
from scoreCheck import getReward, getScore
from distance import get_distance


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear1.bias.data[0] = 300

    def forward(self, distance): # Outputs mean for Gaussian distribution
        distance = self.linear1(distance)
        return distance


model = Policy()


learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

num_episodes = 5000

global variance
initial_variance = 10
final_variance = 1
variance_decay = 500

def trainJump(save_as, curr_checkpoint):
    model.train()
    global variance
    for episode in range(num_episodes):
        variance = final_variance + (initial_variance - final_variance) * \
                        math.exp(-1. * episode / variance_decay)
        print("-----------------------------------------")
        print("Episode:", episode)
        state = get_distance()
        prev_score = getScore()
        print("Distance:", state)
        state = np.array([state])
        state = torch.from_numpy(state)
        state = state.float()
        mean = model(state)
        print("Mean:", float(mean), "Deviation:", float(variance))
        m = Normal(mean, variance)
        action = m.sample()
        print("Action:", action)
        os.system("adb shell input swipe 500 500 500 500 " + str(int(action)))
        time.sleep(0.5)
        reward = getReward(prev_score)
        if reward >= 2:
            reward = 20
        elif reward == 1:
            reward = 0.1
        elif reward < 0:
            time.sleep(1.9)
            os.system("adb shell input tap 550 1700")
            time.sleep(0.4)
        print("Reward:", reward)
        loss = -m.log_prob(action) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Make sure things are fine
        '''
        things = [i for i in range(5)]
        if (episode % 1000) in things:
            print("Episode", episode)
            print("Reward:", actual_reward)
        '''

        if (episode + 1) % 501 == 0:
            save = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            file_name = save_as + str((episode // 1000) + curr_checkpoint) + ".pth"
            torch.save(save, file_name)


save = torch.load('models/PolicyJump1.pth')
model.load_state_dict(save["state_dict"])
optimizer.load_state_dict(save['optimizer'])


trainJump("/Users/alex/PycharmProjects/Wechat_Jump_RL/models/PolicyJump", 1)