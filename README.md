# RNN
>框架图如下。

![rnn框架图(单层)](./img/RNN%E6%A1%86%E6%9E%B6.png)
# 在用RNN做NER任务中数据集长这样(3/rnn_cell/NER)
**940条数据，数据不算多**
```bash
{'text': ['俄', '罗', '斯', '天', '然', '气', '工', '业', '股', '份', '公', '司', '（', 'G', 'a', 'z', 'p', 'r', 'o', 'm', '，', '下', '称', '俄', '气', '）', '宣', '布', '于', '4', '月', '2', '7', '日', '停', '止', '对', '波', '兰', '和', '保', '加', '利', '亚', '的', '天', '然', '气', '供', '应', '。'], 'labels': ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'B-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'O', 'O', 'O', 'B-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
{'text': ['据', 'I', 'T', '之', '家', '消', '息', '，', '台', '湾', '经', '济', '日', '报', '报', '道', '，', '业', '界', '人', '士', '称', '，', '苹', '果', '携', '手', '电', '子', '纸', '（', 'e', 'P', 'a', 'p', 'e', 'r', '）', '龙', '头', '企', '业', '元', '太', '开', '发', '新', '款', 'i', 'P', 'h', 'o', 'n', 'e', '。'], 'labels': ['B-TIM', 'I-TIM', 'B-COU', 'I-COU', 'I-ORG', 'B-LOC', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'B-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-ORG', 'B-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
{'text': ['5', '月', '8', '日', '，', '俄', '罗', '斯', '总', '统', '普', '京', '发', '表', '致', '辞', '，', '向', '白', '俄', '罗', '斯', '、', '亚', '美', '尼', '亚', '、', '摩', '尔', '多', '瓦', '、', '哈', '萨', '克', '斯', '坦', '、', '吉', '尔', '吉', '斯', '斯', '坦', '、', '塔', '吉', '克', '斯', '坦', '、', '土', '库', '曼', '斯', '坦', '、', '乌', '兹', '别', '克', '斯', '坦', '等', '国', '领', '导', '人', '致', '贺', '电', '，', '并', '向', '上', '述', '多', '国', '民', '众', '以', '及', '格', '鲁', '吉', '亚', '和', '乌', '克', '兰', '民', '众', '表', '示', '祝', '贺', '。'], 'labels': ['B-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'O', 'B-COU', 'I-COU', 'I-COU', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
```
# 实验效果（训练集）
```bash
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
```bash
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
```bash
Embedding = nn.Embedding(length_vocab, in_dim)
自实现3层RNN效果
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
torch.nn.RNN效果
all_sentence acc:0.0952, loss:120.9783
all_sentence acc:0.2515, loss:121.5414
all_sentence acc:0.6065, loss:120.8156
all_sentence acc:0.6061, loss:120.8135
all_sentence acc:0.6061, loss:120.8133
all_sentence acc:0.6061, loss:120.8133
all_sentence acc:0.6061, loss:120.8133
all_sentence acc:0.6061, loss:120.8133
all_sentence acc:0.6061, loss:120.8133
all_sentence acc:0.6061, loss:120.8132
```

```bash
one-hot数据格式输入(改一下输入数据的维度in_dim和字典长度一样)
all_sentence acc:0.5592, loss:148.9186
all_sentence acc:0.5663, loss:136.3726
all_sentence acc:0.5685, loss:138.6660
all_sentence acc:0.5752, loss:136.9506
all_sentence acc:0.5663, loss:139.3176
all_sentence acc:0.5660, loss:144.0151
all_sentence acc:0.5700, loss:136.2575
all_sentence acc:0.5636, loss:139.5856
all_sentence acc:0.5638, loss:136.7170
all_sentence acc:0.5596, loss:137.6044
```