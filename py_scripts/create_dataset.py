import nltk
import numpy as np
import re
import itertools
from tqdm import tqdm

class CreateDataset(object):
    def __init__(self, content_list, save_path="list_ngrams.npy"):
        self.alphabets_regex = '^[aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ '
        self.save_path = save_path
        self.content_list = content_list


    def processing(self):
        self.content_list_no_numbers = []
        for text in self.content_list:
            if not self.has_numbers(text):
                self.content_list_no_numbers.append(text)
        self.content_list_no_numbers = [self.preprocessing_data(text) for text in self.content_list_no_numbers]
        # extract phrases
        phrases = itertools.chain.from_iterable(self.extract_phrases(text) for text in self.content_list_no_numbers)
        phrases = [p.strip() for p in phrases if len(p.split()) > 1]

        # gen ngrams
        list_ngrams = []
        for p in tqdm(phrases):
            if not re.match(self.alphabets_regex, p.lower()):
                continue
            if len(phrases) == 0:
                continue

            for ngr in self.gen_ngrams(p, 5):
                if len(" ".join(ngr)) < 40:
                    list_ngrams.append(" ".join(ngr))
        print("DONE extract ngrams, total ngrams: ", len(list_ngrams))
        print(list_ngrams[0:30])

        # save ngrams
        self.save_ngrams(list_ngrams, save_path=self.save_path)

        print("Done create dataset - ngrams")

    def has_numbers(self, inputString):
        return bool(re.search(r'\d', inputString))

    def preprocessing_data(self, row):
        processed = re.sub(
            r'[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ ]',
            "", row)
        return processed

    def extract_phrases(self, text):
        return re.findall(r'\w[\w ]+', text)

    def gen_ngrams(self, text, n=5):
        tokens = text.split()

        if len(tokens) < n:
            return [tokens]

        return nltk.ngrams(text.split(), n)

    def save_ngrams(self, list_ngrams, save_path='list_ngrams.npy'):
        with open(save_path, 'wb') as f:
            np.save(f, list_ngrams)
        print("Saved dataset - ngrams")



