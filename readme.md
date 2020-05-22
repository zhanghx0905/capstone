# Capstone Project

## 文件结构

`data/` 各种数据

`config/` 模型超参数

`src/` 源代码

[word2vec预训练词向量](https://pan.baidu.com/s/1pUqyn7mnPcUmzxT64gGpSw)，生成的词表已经保存在data文件夹下。

## Performance

以测试集准确率为优化指标。

| Model    | Acc    | 注                                                  |
| -------- | ------ | ----------------------------------------------------- |
| fastText | 88.17% | 词袋模型                                              |
| TextCNN  | 91.59% |                |
| DPCNN    | 92.00% |                                                              |
| BiLSTM | 91.58% |                                                       |
| BiLSTM with Attention | 91.60% ||
| TextRCNN | 91.79% | BiLSTM + max pooling                                  |
| BERT  | 94.14% | [来源](https://github.com/ymcui/Chinese-BERT-wwm)，只训练了5个epoch |

除了bert水平都差不多。

TODO：

增加使用2-gram和3-gram信息的fastText.

继续训练bert，这个不着急。

## Usage

```shell
# 在src文件夹下运行命令行
python .\run.py --config .\config\FT.json  # fastText

python .\run.py --config .\config\CNN.json
python .\run.py --config .\config\DPCNN.json  

python .\run.py --config .\config\RNN.json
python .\run.py --config .\config\RNN_Att.json
python .\run.py --config .\config\RCNN.json

python .\run.py --config .\config\BERT.json
```

改变json文件中的`load`和`num_epochs`，可以选择是否加载模型和训练的epoch数。

## 模型介绍

### fastText

[fastText](https://arxiv.org/pdf/1607.01759.pdf)是一种简洁高效的文本分类模型，其思路是将所有输入文本的(也可包含2-gram和3-gram信息)词向量取平均后，经过全连接层输出。虽然简单，但往往比许多深度模型更有效。

![fastText](doc\fastText.png)

### TextCNN

[TextCNN](https://arxiv.org/abs/1408.5882)通过对词向量序列的卷积操作，提取输入文本的2-gram，3-gram乃至n-gram信息，沿着每个卷积核的输出（时间步）做max pooling，用dropout防止过拟合。

卷积层能有效提取局部的语义信息，缺点是捕捉不到长距离关系。

![cnn](doc\cnn.png)

### Deep CNN

在TextCNN的基础上加入了重复多次的池化-卷积-卷积操作，每经过一次池化，序列的长度就缩短一半，这样，越靠上的卷积层就越能提取出序列宏观层面的信息；且因为序列长度的减半，模型消耗的计算资源得到了有效的降低。

![dpcnn](doc\dpcnn.png)

图中的Shallow CNN就是TextCNN.

### BiLSTM

用双向LSTM进行文本分类，取其最后一个时间步上的隐状态过分类器。

在短文本分类中很有效，如果文章很长，这样做会丢失大量中间信息。

![rnn](doc\rnn.png)

### BiLSTM with Attention

为了解决BiLSTM的问题，引入注意力机制。取双向LSTM所有时间步的隐状态输入Self Attention层，将Attention层的输出沿着时间步求和。

![attn](doc\attn.png)

也可以在TextCNN的卷积层前面加self attention，但是我估计不会有质的提升，不做了。

### TextRCNN

结合了BiLSTM和TextCNN的结构。它将双向LSTM的输入和输出拼接在一起，再做max pooling，然后经全连接层输出。

其实就是把TextCNN中的卷积层换成双向LSTM。

![rcnn](doc\rcnn.png)

### BERT

性能上的天花板，原理我还在琢磨。

