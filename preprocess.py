import sys
from data_scripts import *


def sentence_to_token_ids(sentence, vocabulary, tokenizer):
    words = tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words if len(words) <= 80 and len(words) >= 8]


def data_to_token_ids(data_path, vocabulary_path, tokenizer, start=True, lower=False):
    _, vocab = initialize_vocabulary(vocabulary_path, start)
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if lower:
                line = line.lower()
            token_ids = sentence_to_token_ids(line, vocab, tokenizer)
            
            yield " ".join([str(tok) for tok in token_ids])


def merge(tks, tgs, file_path):
    with open(file_path, 'w') as f:
        for x, y in zip(tks, tgs):
            f.write(x + '\t' + y + '\n')


if __name__ == '__main__':
    tokenizer = lambda x: x.split()
    tks = data_to_token_ids('data/train.tks', 'data/wordlist', tokenizer, lower=True)      
    tgs = data_to_token_ids('data/train.tgs', 'data/labellist', tokenizer, False)
    merge(tks, tgs, 'data/train.txt')

