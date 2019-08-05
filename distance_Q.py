import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import math
import numpy as np
import os
from scoreCheck import getReward, getScore
from distance import get_distance

class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(1, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x



final_epsilon = 0.005
initial_epsilon = 0.8
epsilon_decay = 800
global steps_done
steps_done = 0
model = DQN(20)


# Returns node activated and number of milliseconds to hold for
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)
    if sample > eps_threshold:
        print("Exploit")
        with torch.no_grad():
            steps_done += 1
            q_calc = model(state)
            print("Q Vals:", q_calc)
            return int(torch.argmax(q_calc)), int(torch.argmax(q_calc)) * 32 + 300
    else:
        print("Explore")
        node_activated = random.randint(0,19)
        steps_done += 1
        return node_activated, node_activated * 32 + 300


learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


num_episodes = 5000

test_length = 100




def trainJump(save_as, curr_checkpoint):
    model.train()
    for episode in range(num_episodes):
        print("-----------------------------------------")
        print("Episode:", episode)
        state = get_distance() / 4
        prev_score = getScore()
        print("Distance:", state)
        state = np.array([state])
        state = torch.from_numpy(state)
        state = state.float()
        node_activated, action = select_action(state)
        os.system("adb shell input swipe 500 500 500 500 " + str(action))
        time.sleep(0.5)
        actual_reward = getReward(prev_score)
        print("Node Activated:", node_activated, "Action:", action)
        if actual_reward >= 2:
            actual_reward = 10
        elif actual_reward < 0:
            time.sleep(2.1)
            if (getScore() == -10):
                os.system("adb shell input tap 550 1700")
                time.sleep(0.5)
        # memory.push(state, node_activated, actual_reward)
        predicted_reward = model(state).view(20)[node_activated]
        print("Predicted Reward:", float(predicted_reward), "Actual Reward:", actual_reward)
        loss = loss_fn(actual_reward, predicted_reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("-----------------------------------------")

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
            file_name = save_as + str((episode // 500) + curr_checkpoint) + ".pth"
            torch.save(save, file_name)



trainJump("models/distance_q", 0)


'''
Jump:

state = torch.load("Jump7.pth")
model.load_state_dict(state["state_dict"])
optimizer.load_state_dict(state["optimizer"])
'''