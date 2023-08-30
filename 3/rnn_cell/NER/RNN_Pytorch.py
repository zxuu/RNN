import torch
import json
from torch import nn
import torch.utils.data as Data

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

net = nn.RNN(input_size=in_dim, hidden_size=hidden_dim, num_layers=3)
h0 = torch.randn(3*1, 1, hidden_dim)
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
        # texts_embedding = Embedding(torch.tensor(texts_idd))
        # texts_embedding = zero_mat[torch.tensor(texts_idd)]
        input = texts_embedding.unsqueeze(1)    # [seq_len, batch, in_dim]
        labels_y = torch.tensor(labels_idd).view(-1, 1)    # [seq_len, 1]
        optimizer.zero_grad()
        output, h = net(input, h0)    # output:[seq_len, batch, hidden_dim]    h:[num_layers, batch, hidden_dim]

        loss = 0
        is_right = [0]    # 记录一个句子的准确度
        for output_word, label in zip(output, labels_y):  #
            loss = loss + criterion(output_word, label)
            _, idx = output_word.max(dim=1)
            # 记录预测是否正确
            # if label.item()!=26:
            is_right.extend([1 if label.item()==idx.item() else 0])
        sentence_acc = sum(is_right)/len(is_right)
        loss_epoch = loss_epoch + loss
        right.extend([sentence_acc])
        loss.backward()
        optimizer.step()
    print('all_sentence acc:%.4f, loss:%.4f' % (sum(right)/len(right), loss_epoch/len(right)))