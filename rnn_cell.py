from torch import nn
import torch

# 自实现RNN,供理解RNN结构
class RNN_Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(RNN_Cell, self).__init__()
        self.Wx = nn.Parameter(torch.rand(in_dim, hidden_dim))
        self.Wh = nn.Parameter(torch.rand(hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.rand(1, hidden_dim))

    def forward(self, x, h_1):
        """

        Args:
            x (_type_): seq_x: [batch, in_dim]
            h_1 (_type_): [batch, hidden_dim]

        Returns:
            _type_: h
        """
        h = torch.tanh(torch.matmul(x, self.Wx) + torch.matmul(h_1, self.Wh) + self.b)
        return h


class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = RNN_Cell(in_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x (_type_): [seq_len, batch_size, in_dim]

        Returns:
            _type_: _description_
        """
        outs = []
        h = None
        for seq_x in x:
            # seq_x: [batch, in_dim]
            if h is None:    # cell初始输入
                # [batch, hidden_dim]
                h = torch.randn(x.shape[1], self.hidden_dim)
            h = self.rnn_cell(seq_x, h)
            outs.append(torch.unsqueeze(h, 0))
        # cell的每次输出进行拼接
        outs = torch.cat(outs)
        return outs, h


if __name__ == '__main__':
    batch_size, seq_lens, in_dim, hidden_dim = 24, 7, 12, 6
    rnn = RNN(in_dim, hidden_dim)
    x = torch.randn(seq_lens, batch_size, in_dim)
    outs, h = rnn(x)

    print(outs.shape)
    print(h.shape)
