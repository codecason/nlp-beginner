# --- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

# 读取文件
poetrys = []
poetry = ''
with open("poetryFromTang.txt", encoding='utf-8') as f:
    next(f)
    for line in f:
        if len(line) != 1:
            poetry += line.strip('\n')
        else:
            poetrys.append(poetry)
            poetry = ''

# 生成词库
all_word = ''
for potery in poetrys:
    all_word += potery

all_word = all_word.replace('，', '').replace('。', '')

# 统计词频
word_dict = {}

for word in all_word:
    if word not in word_dict:
        word_dict[word] = 1
    else:
        word_dict[word] += 1

word_sort = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
words, _ = zip(*word_sort)

# 获取词典
word_to_token = {word: id for id, word in enumerate(words)}
token_to_word = dict(enumerate(words))


# 将字序列转化为id序列
def transword(char_list):
    ids = [word_to_token.get(char, len(word) - 1) for char in char_list]
    return ids


# ------------------------------------------------------------------------------------------

# 生成数据集，用每句诗的前几个字预测最后一个字。因为每个batch的训练集长度要一致，所以五言诗和七言诗分开。
len1 = 4
len2 = 6
data = [line.replace('，', ' ').replace('。', ' ').split() for line in poetrys]

x_5 = []
x_7 = []
y_5 = []
y_7 = []
for i in data:
    for j in i:
        if len(j) == len1 + 1:
            x_5.append(j[:len1])
            y_5.append(j[-1])
        elif len(j) == len2 + 1:
            x_7.append(j[:len2])
            y_7.append(j[-1])
        else:
            pass

x_5_vec = [transword(i) for i in x_5]
x_7_vec = [transword(i) for i in x_7]
y_5_vec = [transword(i) for i in y_5]
y_7_vec = [transword(i) for i in y_7]

# ------------------------------------------------------------------------------------------
# 定义参数

BATCH_SIZE = 32
learning_rate = 0.01
epoch_num = 100
embedding_size = 300
hidden_size = 256
dropout_size = 0.4
vocab_size = len(all_word)
model_name = 'gru'
num_layers = 2

# 先转换成 torch 能识别的 Dataset
torch_dataset1 = Data.TensorDataset(torch.tensor(x_5_vec, dtype=torch.long),
                                    torch.tensor(y_5_vec, dtype=torch.long))
torch_dataset2 = Data.TensorDataset(torch.tensor(x_7_vec, dtype=torch.long),
                                    torch.tensor(y_7_vec, dtype=torch.long))

# 把 dataset 放入 DataLoader
loader1 = Data.DataLoader(
    dataset=torch_dataset1,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  #
    num_workers=2,  # 多线程来读数据
)

loader2 = Data.DataLoader(
    dataset=torch_dataset2,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  #
    num_workers=2,  # 多线程来读数据
)


# 建立
class PoetryGenerator(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 model_name='lstm'):
        super(PoetryGenerator, self).__init__()
        self.model = model_name
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout=dropout_size)
        self.gru = nn.GRU(embedding_size,
                          hidden_size,
                          num_layers,
                          batch_first=True,
                          dropout=dropout_size)
        self.F = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hs=None):
        x_embedding = self.embed(x)
        batch, seq_len = x.shape
        if hs is None:
            hs = Variable(torch.zeros(num_layers, batch, hidden_size))
        if self.model == 'lstm':
            out, _ = self.lstm(x_embedding, hs)
        else:
            out, _ = self.gru(x_embedding, hs)
        outputs = self.F(out[:, -1, :])

        return outputs, _


model = PoetryGenerator(vocab_size, embedding_size, hidden_size, model_name)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## 训练
from torch.autograd import Variable
# Training
for epoch in range(epoch_num):
    optimizer.zero_grad()
    for step, (batch_x, batch_y) in enumerate(loader1):
        output, _ = model(batch_x)
        loss = criterion(output, batch_y.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    for step, (batch_x, batch_y) in enumerate(loader2):
        output, _ = model(batch_x)
        loss = criterion(output, batch_y.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()


# 从前n个数据随机选
def pick_top_n(preds, top_n=10):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).detach().numpy()
    top_pred_label = top_pred_label.squeeze(0).detach().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)

    return c[0]


def generate_random(max_len=20):
    """自由生成一首诗歌"""
    poetry = []
    sentence_len = 0
    random_word = [np.random.randint(0, vocab_size)]
    _ = Variable(torch.zeros(2, 1, 256))
    input = torch.LongTensor(random_word).reshape(1, 1)
    for i in range(max_len):
        # 前向计算出概率最大的当前词
        proba, _ = model(input, _)
        top_index = pick_top_n(proba)
        char = token_to_word[top_index]

        input = (input.data.new([top_index])).view(1, 1)
        poetry.append(char)
    return poetry


poetry = generate_random()
i = 0
for word in poetry:
    print(word, end='')
    i += 1
    if i % 5 == 0:
        print()
