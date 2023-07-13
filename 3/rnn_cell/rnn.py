import torch

input_size = 4
hidden_size = 4
num_layers = 1
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

# 设置一个one-hot编码的查找，使后续one-hot编码更加便利
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # seg_len * input_size

inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)  # 要把inputs变为seg_len * batch_size * input_size的形式
labels = torch.LongTensor(y_data)  # labels: seg_len * batch_size * 1


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers)

    def forward(self, input):  # 这里的input：seg_len * batch_size * input_size
        # 初始化hidden，即h0: num_layers * batch_size * hidden_size
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)  # out: seg_len * batch_size * hidden_size
        return out.view(-1, self.hidden_size)  # return: seg_len * batch_size 返回一个矩阵


net = Model(input_size, hidden_size, batch_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/5500] loss = %.3f' % (epoch + 1, loss.item()))
