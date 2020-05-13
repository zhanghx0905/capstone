# Capstone Project

### 文件结构

`data/` 数据

`config/` 模型超参数

`src/` 源代码

### 已经有的模型

没有细调参，应该还有一些上升空间。

因为类别均衡，准确率和F1基本相等。

| Model    | Acc    | 备注                                                  |
| -------- | ------ | ----------------------------------------------------- |
| TextCNN  | 91.13% | [参见](https://arxiv.org/abs/1408.5882)               |
| DPCNN    | 91.62% | [参见](https://www.aclweb.org/anthology/P17-1052.pdf) |
| TextRNN  | 91.75% | BiLSTM                                                |
| TextRCNN | 91.63% | BiLSTM + max pooling                                  |
| FastText | 88.14% | 词袋模型                                              |
| Bert     |        | [来源](https://github.com/ymcui/Chinese-BERT-wwm)     |

[预训练词向量](https://pan.baidu.com/s/1pUqyn7mnPcUmzxT64gGpSw)

TODO：

GCN的[论文](https://arxiv.org/pdf/1809.05679.pdf)，似乎效果不错，不知道有没有空复现一下。

我的显卡是1050 Ti，训练Bert非常困难。

作为对比，可以再做一个加入Attention的模型。

最近不想调参。