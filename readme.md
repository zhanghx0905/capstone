# Readme

[下载](https://cloud.tsinghua.edu.cn/d/115cba163e02481e9924/)完整版程序。

## 文件结构

`data/` 各种数据

`config/` 模型超参数

`src/` 源代码

`model/` 训练好的模型和日志

[word2vec预训练词向量](https://pan.baidu.com/s/1pUqyn7mnPcUmzxT64gGpSw)，生成的词表已经保存在data文件夹下。

## 数据来源
[THUNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)的一个子集，从10类新闻中随机抽取了20万条新闻标题，每类各2万条。按照18：1：1划分训练、验证、测试集。


## Performance

以测试集准确率为优化指标。所有的模型都是char-level的。

| Model    | Acc    |                                                   |
| -------- | ------ | ----------------------------------------------------- |
| fastText (bow) | 90.01% |  |
| fastText(2-gram) | 90.01% |  |
| fastText(3-gram) | 92.54% |  |
| TextCNN | 91.48% |  |
| DPCNN    | 92.00% |  |
| BiLSTM | 91.58% |                                                       |
| BiLSTM with Attention | 91.60% ||
| TextRCNN | 91.79% |                                   |
| BERT-wwm-ext | 94.61% | [来源](https://github.com/ymcui/Chinese-BERT-wwm) |
| RoBERTa-wwm-ext | 94.89% |  |

内存有限，GCN的测试仅取数据集的十分之一进行，即20000训练/验证集，1000测试集。

| Model         | Acc    |
| ------------- | ------ |
| fastText(bow) | 85.90% |
| TextGCN       | 83.30% |

## Usage

在src文件夹下运行测试脚本。

```shell
bash test.sh
```

改变json文件中的`load`和`num_epochs`，可以选择是否加载模型和训练的epoch数。

对于gcn，在`text_gcn`文件夹下运行

```
python train.py r8
```

进行训练和测试。

