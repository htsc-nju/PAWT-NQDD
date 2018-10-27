import torch
import torch.nn as nn
# num_layers is the cells
input_size = 10 # sum(t)
sequence_length = 5
batch_size = 1
hidden_size = 20
num_layers = 2
lstm = nn.LSTM(input_size=10, hidden_size=hidden_size, num_layers=num_layers, batch_first = True , bidirectional=False)
input = torch.randn(sequence_length, batch_size, input_size)
h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)
output, (hn, cn) = lstm(input, (h0, c0))
print(cn[-1])
