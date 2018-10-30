import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net import Net
from features_wrap import Feature_Wraper
from quant_gym import Gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
FEATURE_SIZE = 9
WINDOW_SIZE = 10
ACTION_SIZE = 3

MAX_P = 100000


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        # for target updating
        self.learn_step_counter = 0
        # for storing memory
        self.memory_counter = 0
        # initialize memory (state, action, reward, state_)
        self.memory_s = np.zeros((MEMORY_CAPACITY, WINDOW_SIZE, FEATURE_SIZE))
        self.memory_a = np.zeros((MEMORY_CAPACITY, ACTION_SIZE))
        self.memory_r = np.zeros((MEMORY_CAPACITY, 1))
        self.memory_s_ = np.zeros((MEMORY_CAPACITY, WINDOW_SIZE, FEATURE_SIZE))
        self.memory_p = np.zeros(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, features, _actions):
        f_tensor = torch.FloatTensor(features[np.newaxis, :]).cuda()
        a_tensor = torch.FloatTensor(_actions[np.newaxis, :]).cuda()
        action = np.zeros((ACTION_SIZE))
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(f_tensor, a_tensor).cpu()
            action[torch.max(actions_value, 1)[1].data.numpy()] = 1
        else:
            action[np.random.randint(0, ACTION_SIZE)] = 1
        return action

    def store_transition(self, s, a, r, s_, p):
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory_s[index, :, :] = s
        self.memory_a[index] = a
        self.memory_r[index] = r
        self.memory_s_[index, :, :] = s_
        self.memory_p[index] = p
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.argpartition(
            self.memory_p, -BATCH_SIZE)[-BATCH_SIZE:]
        b_s = torch.FloatTensor(self.memory_s[sample_index, :, :]).cuda()
        b_a = torch.FloatTensor(self.memory_a[sample_index, :]).cuda()
        b_al = torch.LongTensor(self.memory_a[sample_index, :]).cuda()
        b_r = torch.FloatTensor(self.memory_r[sample_index, :]).cuda()
        b_s_ = torch.FloatTensor(self.memory_s_[sample_index, :, :]).cuda()

        _, index = torch.max(b_al, dim=1)
        index = index.unsqueeze(1)
        q_eval = self.eval_net(b_s, b_a).gather(1, index)
        q_next = self.target_net(b_s_, b_a).detach()

        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        TD = (q_target - q_eval).detach().cpu().numpy().flatten()
        self.memory_p[sample_index] = np.abs(TD)
        loss = self.loss_func(q_eval, q_target)
        if self.learn_step_counter % 200 == 0:
            print(self.learn_step_counter, "loss = ", loss)
        if self.learn_step_counter % 2000 == 0:
            self.save_model()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.eval_net.state_dict(), "model/eval_net")
        torch.save(self.target_net.state_dict(), "model/target_net")


def run():
    env = Gym()
    dqn = DQN()
    print('\nCollecting experience...')
    for i_episode in range(400):
        s = env.reset()
        _a = np.zeros((3))
        _a[2] = 1
        ep_r = 0
        while True:
            _a = dqn.choose_action(s, _a)
            s_, r, done, no_data = env.step(_a)
            dqn.store_transition(s, _a, r, s_, MAX_P)
            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            if no_data:
                return

            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                break
            s = s_


if __name__ == "__main__":
    run()
