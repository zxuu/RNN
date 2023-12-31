import torch
import json
from torch import nn
import torch.utils.data as Data

# 自实现RNN(多层)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

texts, labels = [], []
unique_text, unique_label = [], []
for line in open('D:\project\\vscode\RNN_Proj\data\\train_data3.txt','r',encoding='utf-8'):
    json_data = json.loads(line)
    texts.append(json_data['text'])
    labels.append(json_data['labels'])
    unique_text.extend(json_data['text'])
    unique_label.extend(json_data['labels'])

unique_text = list(set(unique_text))    # vocab
unique_label = list(set(unique_label))    # unique_label

vocab = unique_text
word2id = {v:k for k,v in enumerate(sorted(vocab))}
id2word = {k:v for k,v in enumerate(sorted(vocab))}
label2id = {v:k for k,v in enumerate(sorted(unique_label))}
id2label = {k:v for k,v in enumerate(sorted(unique_label))}
texts_id = [[word2id[v] for k,v in enumerate(text)] for text in texts]    # [每句话中的每个字在字典中的ID]
labels_id = [[label2id[v] for k,v in enumerate(label)] for label in labels]
length_vocab = len(vocab)
length_labels = len(unique_label)
lenght_texts = len(texts)

in_dim = 512    # 输入维度(字嵌入维度)
hidden_dim = length_labels    # 隐层维度(这里直接和分类数相等了)
epoch = 10
batch_size = 1    # batch=1就不用考虑句子不等长了

word_embedding = nn.Embedding(length_vocab, in_dim)
# torch.rand, torch.randn
# word_embedding = torch.rand(length_vocab, in_dim)   # 输入的词向量是正态分布
# one-hot编码输入
# word_embedding = torch.eye(length_vocab)

class MyDataSet(Data.Dataset):
    def __init__(self, texts, labels):
        super(MyDataSet, self).__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
loader = Data.DataLoader(MyDataSet(texts=texts_id, labels=labels_id), batch_size=1)

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
        h = torch.tanh(self.Wx(x) + self.Wh(h_1))    # [batch, hidden_dim]
        return h

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        """
        input_size: 字符嵌入向量维度
        hidden_size: 隐层维度
        batch_size: 
        """
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell1 = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.rnncell2 = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.rnncell3 = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # self.rnncell1 = RNN_Cell(in_dim=self.input_size, hidden_dim=self.hidden_size)
        self.linear1 = nn.Linear(hidden_dim, in_dim)
        # self.rnncell2 = RNN_Cell(in_dim=self.input_size, hidden_dim=self.hidden_size)
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        # self.rnncell3 = RNN_Cell(in_dim=self.input_size, hidden_dim=self.hidden_size)
        self.linear3 = nn.Linear(hidden_dim, length_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden1, hidden2, hidden3):
        h1 = self.rnncell1(input, hidden1)
        l1 = self.linear1(h1)
        h2 = self.rnncell2(l1, hidden2)
        l2 = self.linear2(h2)
        h3 = self.rnncell3(l2, hidden3)
        l3 = self.linear3(h3)
        y = self.softmax(l3)
        return h1, h2, h3, l3

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size),torch.zeros(self.batch_size, self.hidden_size),torch.zeros(self.batch_size, self.hidden_size)
    



net = RNN(input_size=in_dim, hidden_size=hidden_dim, batch_size=batch_size)
# net.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for i in range(epoch):
    net.train()
    right = []    # 记录全部句子的准确率
    loss_epoch = 0
    for texts_idd, labels_idd in loader:    # 一句话
        texts_idd = texts_idd
        labels_idd = labels_idd
        texts_embedding = word_embedding(torch.tensor(texts_idd))    # [seq_len, in_dim]
        # texts_embedding = Embedding(torch.tensor(texts_idd)).view(-1, batch_size, in_dim)
        # texts_embedding = zero_mat[torch.tensor(texts_idd)]
        labels_y = torch.tensor(labels_idd).view(-1, 1)    # [seq_len, 1]
        loss = 0
        hidden1,hidden2,hidden3 = net.init_hidden()
        is_right = [0]    # 记录一个句子的准确度
        for input, label in zip(texts_embedding, labels_y):  #一句话中的每个词 inputs：seg_len * batch_size * input_size；labels：[seg_len]
            input = input.unsqueeze(0)
            optimizer.zero_grad()
            hidden1,hidden2,hidden3,y = net.forward(input, hidden1=hidden1,hidden2=hidden2,hidden3=hidden3)
            loss = loss + criterion(y, label)  # 要把每个字母的loss累加    =([1,4], [1])
            _, idx = y.max(dim=1)
            # 记录预测是否正确
            # if label.item()!=26:
            is_right.extend([1 if label.item()==idx.item() else 0])
        sentence_acc = sum(is_right)/len(is_right)
        loss_epoch = loss_epoch + loss
        right.extend([sentence_acc])
        loss.backward()
        optimizer.step()
    print('all_sentence acc:%.4f, loss:%.4f' % (sum(right)/len(right), loss_epoch/len(right)))