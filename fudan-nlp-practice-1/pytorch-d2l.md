
#### 动手学深度学习
xavier uniform 初始化

#### 技能
数据预处理

word embedding

模型实现

dnn, rnn, bilstm

seq2seq: 聊天机器人  
ner: 命名实体识别  
bert: 文本相似度  

卷积层  
汇聚层  
全连接层  
时间步  
ReLU  
Sigmoid  
激活函数  
正则化  


高维张量的反向传播

反向传播

$$
\frac{\partial z}{\partial X} = prod(\frac{\partial z}{\partial Y}, \frac{\partial Y}{\partial X})
$$

### CH 6 卷积

#### 6.4. 多输入多输出通道

多个通道用1x1卷积核来调整通道数

思考: 两个卷积核可以用一个卷积核来等效

例如 3通道图片 变成 2通道卷积的输出

#### 6.5. 汇聚层
填充和步幅  
stride 步幅

```pool2d = nn.MaxPool2d(3, padding=1, stride=2)```
最大, 最小, 平均汇聚层

### CH 8
#### 8.3

停词的计算：频率最高的前若干个词

#### 8.5 梯度裁剪
负梯度更新参数

球形梯度投影更新避免学习率过小

