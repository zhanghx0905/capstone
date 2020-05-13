import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
from pytorch_pretrained_bert import BertTokenizer

UNK, PAD = '[UNK]', '[PAD]'  # 未知，padding符号
train_path = "../data/train.txt"
dev_path = "../data/dev.txt"
test_path = "../data/test.txt"
word2vec_path = "../data/sgns.sogou.char"
vocab_path = "../data/vocab.pkl"
trimmed_path = "../data/embedding.npz"
class_list = [x.strip()
              for x in open('..\data\class.txt', encoding='utf8').readlines()]

bert_path = "../data/bert"
CLS = '[CLS]'  # for bert classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenizer(x):
    return [y for y in x]  # per char


def build_vocab(file_path):
    vocab_dic = {}  # word counts
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in f:
            content = line.strip().split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        print(f"words tot: {len(vocab_dic)}")
        vocab_list = sorted([_ for _ in vocab_dic.items()],
                            key=lambda x: x[1], reverse=True)
        vocab_dic = {word_count[0]: idx for idx,
                     word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def load_data(input_path, pad_size):
    texts, labels = [], []
    if os.path.exists(vocab_path):
        word2id = pickle.load(open(vocab_path, 'rb'))
    else:
        word2id = build_vocab(train_path)
        pickle.dump(word2id, open(vocab_path, 'wb'))

    with open(input_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            text, label = line.strip().split('\t')
            labels.append(int(label))

            token = []
            for word in tokenizer(text)[:pad_size]:
                token.append(word2id.get(word, word2id.get(UNK)))
            if len(token) < pad_size:  # padding
                token.extend([word2id[PAD]] * (pad_size - len(token)))
            texts.append(token)

    texts = torch.tensor(texts, dtype=torch.long).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return texts, labels


def load_data_for_bert(input_path, pad_size):
    texts, masks, labels = [], [], []
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            text, label = line.strip().split('\t')
            labels.append(int(label))

            token = tokenizer.tokenize(text)
            token = [CLS] + token
            mask = []
            token_ids = tokenizer.convert_tokens_to_ids(token)

            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
            texts.append(token_ids)
            masks.append(mask)
    texts = torch.tensor(texts, dtype=torch.long).to(device)
    masks = torch.tensor(masks, dtype=torch.long).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return texts, masks, labels


class NewsDataset(data.Dataset):
    def __init__(self, input_path, padding_len, for_bert=False):
        super().__init__()
        self.for_bert = for_bert
        if for_bert:
            self.texts, self.masks, self.labels = load_data_for_bert(
                input_path, padding_len)
        else:
            self.texts, self.labels = load_data(
                input_path, padding_len)
        print(f"loaded {input_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.for_bert:
            return (self.texts[index], self.masks[index]), self.labels[index]
        return self.texts[index], self.labels[index]


def main():
    emb_dim = 300
    if os.path.exists(vocab_path):
        word2id = pickle.load(open(vocab_path, 'rb'))
    else:
        word2id = build_vocab(train_path)
        pickle.dump(word2id, open(vocab_path, 'wb'))

    embeddings = np.random.rand(len(word2id), emb_dim)
    with open(word2vec_path, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip().split(' ')
            if line[0] in word2id:
                idx = word2id[line[0]]
                emb = [float(x) for x in line[1:]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
    np.savez_compressed(trimmed_path, embeddings=embeddings)


if __name__ == "__main__":
    main()
