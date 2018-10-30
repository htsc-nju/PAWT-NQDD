import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
# LSTM Parameters
WINDOW_SIZE = 10
HIDDEN_SIZE = 12
NUM_LAYER = 2
# Gym Parameters
N_ACTIONS = 3
FEATURE_SIZE = 9


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(input_size=FEATURE_SIZE, hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYER, batch_first=True)
        # the action is encodered to one-hot
        self.fc1 = nn.Linear(HIDDEN_SIZE+3, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def init_hidden(self):
        # actualy, it init h & c
        # self.hidden used for choose action, but when we used batches to learn, we need another hidden because of the diff of BATCH_SIZE
        return (torch.zeros(NUM_LAYER, 1, HIDDEN_SIZE).cuda(),
                torch.zeros(NUM_LAYER, 1, HIDDEN_SIZE).cuda())

    def forward(self, features, _actions):
        if len(features) == 1:
            output, self.hidden = self.lstm(features, self.hidden)
        else:
            # hidden will be init to zero
            output, _ = self.lstm(features)
        x = torch.cat((output[:, -1, :], _actions), dim=1)  # 会降维
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


if __name__ == "__main__":
    print('----net test-----')
    n = Net()
    print(n)
