import torch
import json
from torch import nn
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

texts, labels = [], []
unique_text, unique_label = [], []
for line in open('D:\project\\vscode\RNN_Proj\data\\train_data3.txt','r',encoding='utf-8'):
    json_data = json.loads(line)
    texts.append(json_data['text'])
    labels.append(json_data['labels'])
    unique_text.extend(json_data['text'])
    unique_label.extend(json_data['labels'])

unique_text = list(set(unique_text))
unique_label = list(set(unique_label))
# print('len(unique_text),len(unique_label)', len(unique_text), len(unique_label))

vocab = unique_text
word2id = {v:k for k,v in enumerate(sorted(vocab))}
id2word = {k:v for k,v in enumerate(sorted(vocab))}
label2id = {v:k for k,v in enumerate(sorted(unique_label))}
id2label = {k:v for k,v in enumerate(sorted(unique_label))}
texts_id = [[word2id[v] for k,v in enumerate(text)] for text in texts]
labels_id = [[label2id[v] for k,v in enumerate(label)] for label in labels]
length_vocab = len(vocab)
length_labels = len(unique_label)

in_dim = 512
hidden_dim = length_labels
epoch = 10
batch_size = 1


word_embedding = torch.randn(length_vocab, in_dim)   # 输入的词向量是正态分布

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
    
Embedding = nn.Embedding(length_vocab, in_dim)


net = RNN(input_size=in_dim, hidden_size=hidden_dim, batch_size=batch_size)
# net.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for i in range(epoch):
    net.train()
    right = []    # 记录每个句子的准确率
    for texts_idd, labels_idd in loader:
        texts_idd = texts_idd
        labels_idd = labels_idd
        # texts_embedding = word_embedding[torch.tensor(texts_idd)]    # [seq_len, in_dim]
        texts_embedding = Embedding(torch.tensor(texts_idd)).view(-1, batch_size, in_dim)
        labels_y = torch.tensor(labels_idd).view(-1, 1)    # [seq_len, 1]

        loss = 0
        
        hidden = net.init_hidden()
        # print('Predicted string: ', end='')
        is_right = [0]
        for input, label in zip(texts_embedding, labels_y):  # inputs：seg_len * batch_size * input_size；labels：
            # input.cuda()
            # hidden.cuda()
            # label.cuda()
            optimizer.zero_grad()
            hidden = net.forward(input, hidden)
            
            loss = loss + criterion(hidden, label)  # 要把每个字母的loss累加    =([1,4], [1])
            _, idx = hidden.max(dim=1)
            # 输出预测
            # print(id2label[idx.item()]+' ', end='')
            # 记录预测是否正确
            if label.item()!=26:
                is_right.extend([1 if label.item()==idx.item() else 0])
        sentence_acc = sum(is_right)/len(is_right)
        print('sentence acc:%.4f' % (sentence_acc))
        right.extend([sentence_acc])
        loss.backward()
        optimizer.step()
        # print(', Epoch [%d/%d] loss=%.4f' % (i+1,epoch, loss.item()))
    print('all_sentence acc:%.4f' % (sum(right)/len(right)))