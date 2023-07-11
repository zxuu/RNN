import torch
import torch.nn as nn

input_size = 100   # 输入数据编码的维度
hidden_size = 20   # 隐含层维度
num_layers = 4     # 隐含层层数

rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
print("rnn:",rnn)

seq_len = 10        # 句子长度
batch_size = 1      
x = torch.randn(seq_len,batch_size,input_size)        # 输入数据
h0 = torch.zeros(num_layers,batch_size,hidden_size)   # 输入数据

out, h = rnn(x, h0)  # 输出数据

print("out.shape:",out.shape)
print("h.shape:",h.shape)
