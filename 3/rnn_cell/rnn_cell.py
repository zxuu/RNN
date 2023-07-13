import torch
from torch import nn

input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

# 设置一个one-hot编码的查找，使后续one-hot编码更加便利
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # seg_len * input_size

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)  #[5,1,4] 要把inputs变为seg_len * batch_size * input_size的形式
labels = torch.LongTensor(y_data).view(-1, 1)     # [5,1]

# RNNCell
class RNN_Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        """
        in_dim: 4 输入字符的向量的维度
        hidden_dim: 4 隐层维度
        """
        super(RNN_Cell, self).__init__()
        self.Wx = nn.Linear(in_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        # self.b = nn.Parameter(1, hidden_dim)

    def forward(self, x, h_1):
        h = torch.tanh(self.Wx(x) + self.Wh(h_1))
        return h

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        """
        input_size: 4
        hidden_size: 4
        batch_size: 1
        """
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.rnncell = RNN_Cell(in_dim=self.input_size, hidden_dim=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = RNN(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

epoch = 5500
for i in range(epoch):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print('Predicted string: ', end='')
    for input, label in zip(inputs, labels):  # inputs：seg_len * batch_size * input_size；labels：
        hidden = net.forward(input, hidden)
        loss += criterion(hidden, label)  # 要把每个字母的loss累加    =([1,4], [1])
        _, idx = hidden.max(dim=1)
        # 输出预测
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/%d] loss=%.4f' % (i+1,epoch, loss.item()))
