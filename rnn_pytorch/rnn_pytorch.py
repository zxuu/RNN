import torch
from torch import nn


# 调包
# rnn = nn.RNN(input_size=12, hidden_size=6, batch_first=True)    # 默认为false
rnn = nn.RNN(input_size=12, hidden_size=6)

input = torch.randn(24, 5, 12)
outputs, hn = rnn(input)

print(outputs.size())
print(hn.size())
