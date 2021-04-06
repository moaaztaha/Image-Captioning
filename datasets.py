import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


from PIL import Image

import pyarabic.araby as araby
import pandas as pd
import spacy

spacy_en = spacy.load('en')


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transforms=None, freq_threshold=2, split='train'):
        # split has to be one of {train, val, test}
        assert split in {'train', 'val', 'test'}

        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.vocab = vocab
        self.transforms = transforms

        # getting image ids and captions based on split
        self.df = self.df[self.df['split'] == split]
        self.img_idx = self.df['file_name'].values
        self.captions = self.df['caption'].values

        # printing some info
        print(f"Dataset split: {split}")
        print(f"Unique Image: {self.df.file_name.nunique()}")
        print(f"Size: {self.df.shape[0]}")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.img_idx[index]
        img = Image.open(self.root_dir+img_id).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transforms.ToTensor()(img)

        # numericalize captions
        word_idx = self.vocab.numericalize(caption)
        caption = [self.vocab.stoi["<sos>"]]
        caption.extend(word_idx)
        caption.append(self.vocab.stoi["<eos>"])
        return img, torch.tensor(caption)


class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize_en(text, lang='en'):
        if lang == 'ar':
            return [tok for tok in araby.tokenize(text.lower())]
        else:
            return [tok.text for tok in spacy_en.tokenizer(text.lower())]

    def build_vocabulary(self, sentence_list, lang='en'):
        freqs = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize_en(sentence, lang):
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


class collate_fn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        captions = [item[1] for item in batch]
        captions = pad_sequence(
            captions, batch_first=False, padding_value=self.pad_idx)

        return images, captions


def get_loaders(bs, images_path, df_path, transform, vocab):
    pad_idx = vocab.stoi['<pad>']

    train_loader = DataLoader(
        dataset=FlickrDataset(images_path, df_path,
                              transforms=transform, vocab=vocab, split='train'),
        batch_size=bs,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn(pad_idx),
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=FlickrDataset(images_path, df_path,
                              transforms=transform, vocab=vocab, split='val'),
        batch_size=bs,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn(pad_idx),
        drop_last=True
    )

    return train_loader, valid_loader
