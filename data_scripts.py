import random
import numpy as np


PAD_TOKEN = '<pad>' # pad symbol
UNK_TOKEN = '<unk>' # unknown word

# we always put them at the start.
_START_VOCAB = [PAD_TOKEN, UNK_TOKEN]
PAD_ID = 0
UNK_ID = 1


def initialize_vocabulary(vocab_path, start=True):
    """
    """
    with open(vocab_path, 'r', encoding="utf-8") as f:
        rev_vocab = [line.strip() for line in f]
    if start:
        rev_vocab = _START_VOCAB + rev_vocab
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return rev_vocab, vocab


def load_data(file_path, training=False):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        content = [line.strip() for line in lines if len(line.strip()) > 1]
        data = [(line.split('\t')[0].strip().split(), line.split('\t')[1].strip().split()) for line in content]
        if training:
            data = [([int(item) for item in sent], [int(item) for item in tag]) for sent, tag in data if len(sent) < 80 and len(sent) > 8]
        else:
            data = [([int(item) for item in sent], [int(item) for item in tag]) for sent, tag in data]
    return data


def load_pretrain(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        pretrain_embedding = []
        
        for line in lines:
            embed_word = line.strip().split(',')
            embed_word = [float(item) for item in embed_word]
            pretrain_embedding.append(embed_word)
            
        pretrain_embed_dim = len(pretrain_embedding[0])
        tmp = []
        tmp.append([random.uniform(-1, 1) for _ in range(pretrain_embed_dim)])
        tmp.append([random.uniform(-1, 1) for _ in range(pretrain_embed_dim)])
        pretrain_embedding = tmp + pretrain_embedding   
    return np.matrix(pretrain_embedding)