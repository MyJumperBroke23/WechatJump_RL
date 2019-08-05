import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import math
from collections import namedtuple
import os
from scoreCheck import getReward, getScore
from preprocess import getProcessedImage, getLessProcessed, getImage


class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.linear = nn.Linear(950400, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.linear(x)


final_epsilon = 0.005
initial_epsilon = 0.05
epsilon_decay = 2000
global steps_done
steps_done = 0
model = DQN(16)


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
            return int(torch.argmax(q_calc)), int(torch.argmax(q_calc)) * 50 + 300
    else:
        print("Explore")
        node_activated = random.randint(0,15)
        steps_done += 1
        return node_activated, node_activated * 50 + 300


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = ReplayMemory(128)

learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
BATCH_SIZE = 32

num_episodes = 5000


def optimize_model():
    if len(memory) < BATCH_SIZE:
        time.sleep(3)
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.Tensor(batch.action)
    action_batch = action_batch.type(torch.LongTensor)
    reward_batch = torch.Tensor(batch.reward)
    predicted_reward_batch = model(state_batch)
    predicted_reward_batch = torch.gather(predicted_reward_batch, 1, action_batch.unsqueeze(1))
    # print("Actions:", action_batch, "\nRewards:", reward_batch, "\nPredicted Rewards:", predicted_reward_batch)

    loss = loss_fn(reward_batch, predicted_reward_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def trainJump(save, save_as=None, curr_checkpoint=None):
    model.train()
    for episode in range(num_episodes):
        prev_score = getScore()
        print("-----------------------------------------")
        print("Episode:", episode)

        # Get state and select action based on state
        state = torch.Tensor(getLessProcessed()).unsqueeze(0)
        state_shape = state.shape
        state = state.view(state_shape[0], state_shape[3], state_shape[1], state_shape[2])
        node_activated, action = select_action(state)
        os.system("adb shell input swipe 500 500 500 500 " + str(action))

        optimize_model()  # Optimize here to save time

        # Get actual reward and push state, node_activated, and actual_reward to memory
        actual_reward = getReward(prev_score)
        print("Node Activated:", node_activated, "Action:", action)
        if actual_reward >= 2:
            actual_reward = 10
        elif actual_reward < 0:
            onDeath()
        memory.push(state, node_activated, actual_reward)

        # Print predicted and actual rewards
        predicted_reward = model(state).view(16)[node_activated]
        print("Predicted Reward:", float(predicted_reward), "Actual Reward:", actual_reward)
        print("-----------------------------------------")

        if save:
            if (episode + 1) % 1001 == 0:
                save_file = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                file_name = save_as + str((episode // 1000) + curr_checkpoint) + ".pth"
                torch.save(save_file, file_name)


def onDeath():
    time.sleep(1.9)
    if getScore() == -10:
        os.system("adb shell input tap 550 1700")
        time.sleep(0.5)
    else:  # Sometimes a weird pop up will come up, this closes it
        image = getImage()
        if (image[1000, 200] == [255, 255, 255]).all():
            os.system("adb shell input tap 200 1400")
            time.sleep(0.1)
            os.system("adb shell input tap 550 1700")
