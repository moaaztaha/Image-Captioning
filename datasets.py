import pandas as pd
import spacy

spacy_en = spacy.load('en')


class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text.lower())]

    def build_vocabulary(self, sentence_list):
        freqs = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize_en(sentence):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1

                if freqs[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize_en(text)

        return [self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
                for token in tokenized_text]


def build_vocab(data_file, freq_threshold=2, split='train'):
    df = pd.read_csv(data_file)
    df = df[df['split'] == split]
    captions = df.caption.values

    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(captions)

    return vocab
