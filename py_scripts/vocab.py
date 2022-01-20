from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm.notebook import tqdm
import numpy as np
import pickle


class VocabChar(object):
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2

        self.chars = chars

        self.i2c = {i + 3: c for i, c in enumerate(chars)}

        self.c2i = {c: i + 3 for i, c in enumerate(chars)}

        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]

    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent

    def __len__(self):
        return len(self.c2i) + 3

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars


class VocabWord(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.go = 1
        self.eos = 2

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def encode(self, chars):
        encode_sent = []
        encode_sent.append(self.go)
        for word in word_tokenize(chars):
            if not word in self.word2idx:
                encode_sent.append(self.word2idx['<unk>'])
            else:
                encode_sent.append(self.word2idx[word])
        encode_sent.append(self.eos)
        while len(encode_sent) < 10:
            encode_sent.append(0)

        return encode_sent

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]


def build_vocab(all_clean_captions):
    captions = []
    counter = Counter()
    for caption in tqdm(all_clean_captions):
        captions.append(caption)
        tokens = word_tokenize(caption)
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt>=2]
    vocab = VocabWord()
    vocab.add_word('<pad>')  # 0
    vocab.add_word('<start>')  # 1
    vocab.add_word('<end>')  # 2
    vocab.add_word('<unk>')  # 3
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


if __name__ == "__main__":
    train_captions = np.load('../noob/train_normal_captions.npy')
    vocab = build_vocab(train_captions)
    print(f"Length of vocab {len(vocab)}")
    with open('../vocab.pkl', 'wb') as outp:
        pickle.dump(vocab, outp, pickle.HIGHEST_PROTOCOL)
    with open('../vocab.pkl', 'rb') as inp:
        vocab = pickle.load(inp)
    print(f"Length of vocab {len(vocab)}")
