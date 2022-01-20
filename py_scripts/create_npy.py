import random
import numpy as np
import torch
from py_scripts.create_dataset import CreateDataset
MAX_LEN = 40

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(seed=1)

if __name__ == "__main__":
    with open('../train_sentence.txt', 'r', encoding='utf-16') as f:
        train_data = f.read().split('\n')
    print(train_data[0])
    print(f"Number of sentences in train {len(train_data)}")
    with open('../valid_sentence.txt', 'r', encoding='utf-16') as f:
        valid_data = f.read().split('\n')
    with open('../test_sentence.txt', 'r', encoding='utf-16') as f:
        test_data = f.read().split('\n')
    # data = list(itertools.chain.from_iterable([gen_ngrams(item) for item in data]))
    # print(f"First 10 ngrams: {data[:10]}")
    # print(f"Length ngrams {len(data)}")
    # save_ngrams(data)
    create = CreateDataset(train_data, save_path = 'list_5gram_new_nonum_train.npy')
    create.processing()
    create = CreateDataset(valid_data, save_path='list_5gram_new_nonum_valid.npy')
    create.processing()
    create = CreateDataset(test_data, save_path='list_5gram_new_nonum_test.npy')
    create.processing()
    # list_ngrams_train = np.load('list_ngrams_train.npy')
    # list_ngrams_valid = np.load('list_ngrams_valid.npy')
    # list_ngrams_test = np.load('list_ngrams_test.npy')
    # synthesizer = SynthesizeData(word_tokenize)
    # ds_train = SpellingDataset(list_ngrams_train, synthesizer, vocab, 40)
    # ds_valid= SpellingDataset(list_ngrams_valid, synthesizer, vocab, 40)
    # ds_test = SpellingDataset(list_ngrams_test, synthesizer, vocab, 40)
    # train_loader = DataLoader(ds_train, batch_size=200, shuffle=True)
    # val_loader = DataLoader(ds_valid, batch_size=200)
    # test_loader = DataLoader(ds_test, batch_size=200)
    # print(len(train_loader), len(val_loader), len(test_loader))
