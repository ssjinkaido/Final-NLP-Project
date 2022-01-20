import codecs
from itertools import islice
import random

# num_lines = sum(1 for line in open('corpus-title.txt', encoding="utf8"))
# print(f"Number of lines in text file:{num_lines}")


def file_read_from_head(fname, nlines):
    with open(fname, encoding="utf8") as f:
        for line in islice(f, nlines):
            print(line)


def save_file_to_list(fname):
    with open(fname, encoding="utf8") as f:
        file_content = f.read()
        content_list = file_content.split("\n")

    return content_list


def random_content_file(content_list):
    content_list = [d for d in content_list if 30 < len(d) < 500]
    random.shuffle(content_list)
    print(f"First 5 lines: {content_list[:5]}")
    print(f"Length: {len(content_list)}")
    return content_list


def train_valid_test_split(content_list):
    split_train = 800000
    split_valid = 10000
    train_list = content_list[:split_train]
    valid_list = content_list[split_train:split_train + split_valid]
    test_list = content_list[split_train + split_valid:split_train + split_valid*2]
    print(f"Length train set {len(train_list)} \n")
    print(f"Length valid set {len(valid_list)} \n")
    print(f"Length test set {len(test_list)} \n")
    return train_list, valid_list, test_list


def write_train_test_file(train_list, valid_list, test_list):
    with open('../train_sentence.txt', 'w', encoding='utf-16') as outfile:
        train_data = '\n'.join(train_list)
        outfile.write(train_data)
        print('Finish writing train file')

    with open('../valid_sentence.txt', 'w', encoding='utf-16') as outfile:
        test_data = '\n'.join(valid_list)
        outfile.write(test_data)
        print('Finish writing valid file')

    with open('../test_sentence.txt', 'w', encoding='utf-16') as outfile:
        test_data = '\n'.join(test_list)
        outfile.write(test_data)
        print('Finish writing test file')


file_read_from_head('corpus-title.txt', 10)
content_list = save_file_to_list('corpus-title.txt')
print(f"First 5 lines{content_list[:5]} \n")
list_shuffle = random_content_file(content_list)
train_list, valid_list, test_list = train_valid_test_split(list_shuffle)
write_train_test_file(train_list, valid_list, test_list)