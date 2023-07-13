# RNN
```python
word_embedding = torch.rand(length_vocab, in_dim)   # 输入的词向量是均匀分布
```
均匀分布输入准确率：all_sentence acc:0.0577

```python
word_embedding = torch.randn(length_vocab, in_dim)   # 输入的词向量是正态分布
```
正态分布输入准确率：all_sentence acc:0.1174

```python
Embedding = nn.Embedding(length_vocab, in_dim)
...
texts_embedding = Embedding(torch.tensor(texts_idd)).view(-1, batch_size, in_dim)
```
nn.embedding输入准确率：all_sentence acc:0.1734