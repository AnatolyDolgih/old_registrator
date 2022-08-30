import torch
import re
import spacy

from progress.bar import ChargingBar 
from os.path import exists
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from typing import List

BOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
UNK_IDX = 3

class MyDataset(Dataset):
    def __init__(self, list):
        self.data = list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def load_tokenizers():
    spacy_ru = spacy.load("ru_core_news_sm")
    return spacy_ru

spacy_ru = load_tokenizers()
    
def normString(str):
    str = str.lower().rstrip().lstrip()
    str = re.sub(r"([.!?])", r" \1", str)
    str = re.sub(r"[^а-яА-Я.!?]+", r" ", str)
    return str

def readTrainData(data_path):
    bar = ChargingBar('Обработка тренировочных данных', max = 8000)
    lines = []
    for i in range(1, 8001):
        with open(data_path + f"log_appraisals{i}.txt", "r", encoding='cp1251') as f:
            for line in f:
              lines.append(line.split(":")[0])
            lines.pop(len(lines) - 1)
        bar.next()
    data = [[normString(lines[i]), normString(lines[i+1])] for i in range(0, len(lines), 2)]
    bar.finish()
    return data

def readValidData(data_path):
    bar = ChargingBar('Обработка валидационных данных', max = 2000)
    lines = []
    for i in range(8001, 10001):
        with open(data_path + f"log_appraisals{i}.txt", "r", encoding='cp1251') as f:
            for line in f:
              lines.append(line.split(":")[0])
            lines.pop(len(lines) - 1)
        bar.next()
    data = [[normString(lines[i]), normString(lines[i+1])] for i in range(0, len(lines), 2)]
    bar.finish()
    return data

def tokenize_ru(text):
        return tokenize(text, spacy_ru)
    
def build_vocabulary(spacy_ru, train_data,
                     valid_data):

    print("Составление словаря посетителя ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train_data + valid_data, tokenize_ru, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    print(type(vocab_src))

    print("Составление словаря регистратора ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train_data + valid_data, tokenize_ru, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    print(type(vocab_tgt))

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

def load_vocab(spacy_ru, train_data, valid_data):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_ru, train_data, valid_data)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Составление закончено.\nРазмеры словарей:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def get_vocab_size(path):
    vocab_src, vocab_tgt = torch.load(path)
    return len(vocab_src), len(vocab_tgt)