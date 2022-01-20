import torch
from torch.utils.data import Dataset
import numpy as np
class SpellingDataset(Dataset):
    def __init__(self, list_ngram, synthesize, vocab, maxlen):
        self.list_ngram = list_ngram
        self.synthesize = synthesize
        self.vocab = vocab
        self.max_len = maxlen

    def __getitem__(self, index):
        train_target = self.list_ngram[index]
        train_text = self.synthesize.add_noise(train_target)

        train_text_encode = self.vocab.encode(train_text)
        train_target_encode = self.vocab.encode(train_target)

        train_text_length = len(train_text_encode)
        train_target_length = len(train_target_encode)

        mask = [1] * train_text_length

        if (train_text_length < self.max_len):
            pad_length = self.max_len - train_text_length
            train_text_encode = np.array(train_text_encode)
            train_text_encode = np.concatenate((train_text_encode, np.zeros(pad_length)), axis=0)
            mask = mask+([0]*pad_length)

        elif (train_text_length >= self.max_len):
            train_text_encode = train_text_encode[0:self.max_len]
            train_text_encode = np.array(train_text_encode)


        if (train_target_length < self.max_len):
            pad_length = self.max_len - train_target_length
            train_target_encode = np.array(train_target_encode)
            train_target_encode = np.concatenate((train_target_encode, np.zeros(pad_length)), axis=0)

        elif (train_target_length >= self.max_len):
            train_target_encode = train_target_encode[0:self.max_len]
            train_target_encode = np.array(train_target_encode)

        tensor_text = torch.from_numpy(train_text_encode)
        tensor_target = torch.from_numpy(train_target_encode)
        mask = torch.tensor(mask, dtype=torch.long)
        return tensor_text, tensor_target, mask

    def __len__(self):
        return len(self.list_ngram)