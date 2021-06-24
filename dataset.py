import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torchvision.transforms as transfroms

from PIL import Image
import pandas as pd
from tqdm import tqdm
import pickle


class CaptionDataset(Dataset):
    """ 
    Caption Dataset Class
    """

    def __init__(self, imgs_dir, captions_file, vocab, transforms=None, split='train'):
        """
        :param imgs_dir: folder where images are stored
        :param captions_file: the df file with all caption information
        :param vocab: vocabuary object
        :param transforms: image transforms pipeline
        :param split: data split
        """

        # split has to be one of {'train', 'val', 'test'}
        assert split in {'train', 'val', 'test'}

        self.imgs_dir = imgs_dir
        self.df = pd.read_json(captions_file)
        self.df = self.df[self.df['split'] == split]
        self.vocab = vocab
        self.transforms = transforms
        self.split = split

        self.dataset_size = self.df.shape[0]
        # printing some info
        print(f"Dataset split: {split}")
        print(f"Unique images: {self.df.file_name.nunique()}")
        print(f"Total size: {self.dataset_size}")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):

        # loading the image
        img_id = self.df['file_name'].values[index]
        img = Image.open(self.imgs_dir+img_id).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transfroms.ToTensor()(img)

        # loading current caption
        cap_len = self.df['tok_len'].values[index] + 2 # <sos> and <eos>
        tokens = self.df['tokens'].values[index]
        caption = torch.LongTensor(self.vocab.numericalize(tokens, cap_len))

        if self.split is 'train':
            return img, caption, cap_len
        else:
            # for val and test return all captions for calculate the bleu scores
            captions_tokens = self.df[self.df['file_name'] == img_id].tokens.values
            captions_lens = self.df[self.df['file_name'] == img_id].tok_len.values
            all_tokens = []
            for token, cap_len in zip(captions_tokens, captions_lens):
                all_tokens.append(self.vocab.numericalize(token, cap_len)[1:]) # remove <sos>

            return img, caption, cap_len, torch.tensor(all_tokens)


def build_vocab(data_file, freq_threshold=2):
    df = pd.read_json(data_file)
    # if split:
    #     df = df[df['split'] == split]

    tokens = df.tokens.values

    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(tokens)

    return vocab

def top10k_vocab(words_list):
    
    tokens = [words_list]
    
    vocab = Vocabulary(1) # threshold = 0 to include all
    vocab.build_vocabulary(tokens)
    return vocab

def get_10k_vocab(path="10k_words.txt"):
    # loading words list 
    with open(path, "rb") as f:
        top10kwords = pickle.load(f)
    
    return top10kwords
    




class Vocabulary:
    def __init__(self, freq_threshold=2, max_len=100):
        self.freq_threshold = freq_threshold
        self.max_len = max_len
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}

    def __len__(self):
        return len(self.itos)

    # @staticmethod
    # def tokenize_en(text):
    #     return [tok.text for tok in spacy_en.tokenizer(text.lower())]


    def build_vocabulary(self, tokens_list):
        freqs = {}
        idx = 4

        for tokens in tqdm(tokens_list):
            for word in tokens:
                # print(word)
                # break
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1

                if freqs[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, tokens, cap_len):
        return [self.stoi['<sos>']] + [self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
                                       for token in tokens] + [self.stoi['<eos>']] + [self.stoi['<pad>']] * (self.max_len - cap_len)

    def indextostring(self, idx):
        sent_tokens = []
        for sent in idx:
            sent_tokens.append([self.itos[i] for i in sent if i not in {self.stoi['<sos>'], self.stoi['<pad>'], self.stoi['<eos>']}])
        return sent_tokens

# class collate_fn:
#     def __init__(self, pad_idx, split='train'):
#         self.pad_idx = pad_idx
#         self.split = split

#     def __call__(self, batch):
#         images = [item[0].unsqueeze(0) for item in batch]
#         images = torch.cat(images, dim=0)

#         captions = [item[1] for item in batch]
#         captions = pad_sequence(
#             captions, batch_first=False, padding_value=self.pad_idx)

#         cap_lens = [item[2] for item in batch]
#         cap_lens = torch.cat(cap_lens, dim=0)

#         if self.split == 'train':
#             return images, captions, cap_lens
#         else:
#             all_caps = [item[3] for item in batch]
#             all_caps
        

#         return images, captions


def get_loaders(bs, images_path, df_path, transform, vocab, test=False, n_workers=0):
    #pad_idx = vocab.stoi['<pad>']

    if test:
        test_loader = DataLoader(
            dataset=CaptionDataset(images_path, df_path,
                                    transforms=transform, vocab=vocab, split='test'),
            batch_size=bs,
            num_workers=n_workers,
            shuffle=True,
            pin_memory=True
        )
        return test_loader



    train_loader = DataLoader(
        dataset=CaptionDataset(images_path, df_path,
                              transforms=transform, vocab=vocab, split='train'),
        batch_size=bs,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=CaptionDataset(images_path, df_path,
                              transforms=transform, vocab=vocab, split='val'),
        batch_size=bs,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=True,
    )

    return train_loader, valid_loader