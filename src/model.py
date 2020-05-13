import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import trimmed_path, bert_path, class_list
from pytorch_pretrained_bert import BertModel

embedding_pretrained = torch.tensor(
    np.load(trimmed_path)["embeddings"].astype('float32'))
class_num = len(class_list)


class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        args = {
            'hidden_size': 768,
        }
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(args['hidden_size'], class_num)

    def forward(self, x):
        context, mask = x[0], x[1]
        _, pooled = self.bert(context, attention_mask=mask,
                              output_all_encoded_layers=False)
        x = self.fc(pooled)
        return x


class DPCNN(nn.Module):
    def __init__(self, embedding_len):
        super().__init__()
        args = {
            'out_channels': 250
        }
        self.conv_region = nn.Conv2d(1, args['out_channels'], (3, embedding_len), padding=(1, 0))
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(args['out_channels'], args['out_channels'], 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(args['out_channels'], args['out_channels'], 3, padding=1))
        self.embedding = nn.Embedding.from_pretrained(
            embedding_pretrained, freeze=False)
        self.fc = nn.Linear(args['out_channels'], class_num)
        self.max_pool = nn.MaxPool1d(3, 2)

    def forward(self, texts):
        embedding = self.embedding(texts)
        x = embedding.unsqueeze(1)
        px = self.conv_region(x).squeeze(3)  # (bs, oc, len)

        x = self.conv2(px)
        x = px + x

        for _ in range(2):
            px = self.max_pool(x)
            x = self.conv2(px)
            x = px + x
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # (bs, oc)
        x = self.fc(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, embedding_len):
        super().__init__()
        args = {
            'out_channels': 256,
            'kernal_size': [2, 3, 4],
            'dropout': 0.5
        }
        self.convs = nn.ModuleList([
            nn.Conv2d(1, args['out_channels'], (ks, embedding_len))
            for ks in args['kernal_size']
        ])
        self.fc = nn.Linear(args['out_channels']*len(args['kernal_size']), class_num)
        self.dropout = nn.Dropout(args['dropout'])
        self.embedding = nn.Embedding.from_pretrained(
            embedding_pretrained, freeze=False)

    def conv_and_pool(self, conv_layer, x):
        x = F.relu(conv_layer(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, texts):
        embedding = self.embedding(texts)
        x = embedding.unsqueeze(1)  # (bs, 1, padding, embedding)
        x = [self.conv_and_pool(conv, x)
             for conv in self.convs]  # (bs, oc) * len(ks)
        x = torch.cat(x, 1)  # (bs, oc*len(ks))
        x = self.dropout(x)
        x = self.fc(x)
        return x


class TextRCNN(nn.Module):
    def __init__(self, embedding_len):
        super().__init__()
        self.args = {
            'hidden': 256,
            'num_layers': 1,
        }
        self.lstm = nn.LSTM(input_size=embedding_len,
                            hidden_size=self.args['hidden'],
                            num_layers=self.args['num_layers'],
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(self.args['hidden'] * 2 + embedding_len, class_num)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_pretrained, freeze=False)

    def forward(self, texts):
        embedding = self.embedding(texts)
        states, _ = self.lstm(embedding)  # bs, padding, hidden*2
        x = torch.cat((states, embedding), 2)  # bs, padding, hidden*2+embedding
        x = F.relu(x)
        x, _ = torch.max(x, dim=1)  # bs, hidden*2+embedding
        return self.fc(x)


class TextRNN(nn.Module):
    def __init__(self, embedding_len):
        super().__init__()
        args = {
            'hidden': 128,
            'num_layers': 2,
            'dropout': 0.5
        }
        self.lstm = nn.LSTM(input_size=embedding_len,
                            hidden_size=args['hidden'],
                            num_layers=args['num_layers'],
                            bidirectional=True,
                            batch_first=True,
                            dropout=args['dropout'])
        self.fc = nn.Linear(args['hidden'] * 2, class_num)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_pretrained, freeze=False)

    def forward(self, texts):
        embedding = self.embedding(texts)
        x, _ = self.lstm(embedding)
        x = self.fc(x[:, -1, :])  # bs, hidden*2
        return x


class FastText(nn.Module):
    def __init__(self, embedding_len, padding_len):
        super().__init__()
        self.avg = nn.AvgPool1d(padding_len)
        self.fc = nn.Linear(embedding_len, class_num)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_pretrained, freeze=False)

    def forward(self, texts):  # batch, padding, embedding
        embedding = self.embedding(texts)
        x = torch.transpose(embedding, 1, 2)  # batch, embedding, padding
        x = self.avg(x)  # batch, embedding, 1
        x = self.fc(x.squeeze(2))
        return x
