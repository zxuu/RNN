# RNN
```python
word_embedding = torch.rand(length_vocab, in_dim)   # 输入的词向量是均匀分布

all_sentence acc:0.5596, loss:150.2438
all_sentence acc:0.5693, loss:138.5587
all_sentence acc:0.5642, loss:144.7185
all_sentence acc:0.5767, loss:137.9415
all_sentence acc:0.5697, loss:139.0065
all_sentence acc:0.5706, loss:138.2792
all_sentence acc:0.5720, loss:140.5094
all_sentence acc:0.5646, loss:139.6325
all_sentence acc:0.5754, loss:138.4563
all_sentence acc:0.5670, loss:139.1973
```
```python
word_embedding = torch.randn(length_vocab, in_dim)   # 输入的词向量是正态分布
all_sentence acc:0.5624, loss:109.9485
all_sentence acc:0.5760, loss:103.4124
all_sentence acc:0.5785, loss:102.8813
all_sentence acc:0.5762, loss:103.7379
all_sentence acc:0.5853, loss:101.9712
all_sentence acc:0.5867, loss:102.2876
all_sentence acc:0.5842, loss:104.6158
all_sentence acc:0.5900, loss:101.7214
all_sentence acc:0.5878, loss:102.6558
all_sentence acc:0.5885, loss:102.9117
```
```python
Embedding = nn.Embedding(length_vocab, in_dim)
all_sentence acc:0.5706, loss:108.1364
all_sentence acc:0.5827, loss:105.3522
all_sentence acc:0.5825, loss:102.8504
all_sentence acc:0.5776, loss:103.5315
all_sentence acc:0.5784, loss:103.9420
all_sentence acc:0.5776, loss:103.2978
all_sentence acc:0.5721, loss:105.8887
all_sentence acc:0.5744, loss:104.0500
all_sentence acc:0.5735, loss:104.3058
all_sentence acc:0.5709, loss:104.7786
```